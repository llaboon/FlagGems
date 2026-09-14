# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
from triton import language as tl

from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import c_contiguous_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


@triton.jit
def repeat_interleave_self_int_kernel(
    in_ptr,
    out_ptr,
    numel,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    IS3: tl.constexpr,
    IS4: tl.constexpr,
    IS5: tl.constexpr,
    ISD: tl.constexpr,
    OSD: tl.constexpr,
    SDR: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program handles BLOCK consecutive *output* elements (contiguous
    # stores, which the XPU backend can vectorize); dims beyond ndim are size
    # 1 / stride 0 so a fixed 6-wide chain covers any input rank.
    # p is decomposed by the OUTPUT shapes (S0..S5 = S0..S_{n-1}, padded with
    # 1). The input offset of output index (i_0..i_{n-1}) is
    #   sum_j i_j * IS_j + (i_D // R - i_D) * ISD
    # with i_D = (p // OSD) % SDR (SDR = size of the output along `dim`,
    # OSD = the output's c-contiguous stride along `dim`).
    pid = ext.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    m = p < numel
    t = p
    i5 = t % S5
    t = t // S5
    i4 = t % S4
    t = t // S4
    i3 = t % S3
    t = t // S3
    i2 = t % S2
    t = t // S2
    i1 = t % S1
    t = t // S1
    i0 = t
    base = i0 * IS0 + i1 * IS1 + i2 * IS2 + i3 * IS3 + i4 * IS4 + i5 * IS5
    iD = (p // OSD) % SDR
    in_off = base + (iD // R - iD) * ISD
    val = tl.load(in_ptr + in_off, mask=m)
    tl.store(out_ptr + p, val, mask=m)


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_KUNLUNXIN REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    if (not inp.is_contiguous()) and len(inp_shape) <= 6:
        # Non-contiguous inputs: the broadcast copy below lowers the 0-stride
        # view into the XPU tiled local-memory path (gm2lm/lm2gm with a
        # 512-thread cluster), which miscompiles on XPU for some non-contiguous
        # shapes (illegal memory access / wrong values). Use a dedicated kernel
        # that iterates the output space with contiguous stores instead.
        n = len(inp_shape)
        is_pad = (inp_stride + [0] * (6 - n))[:6]
        o_strides = c_contiguous_stride(output_shape)
        osd = o_strides[dim]
        sdr = output_shape[dim]
        numel = output.numel()
        BLOCK = 2048
        grid = (triton.cdiv(numel, BLOCK),)
        repeat_interleave_self_int_kernel[grid](
            inp,
            output,
            numel,
            *((output_shape + [1] * (6 - n))[:6])[1:],
            *is_pad,
            is_pad[dim],
            osd,
            sdr,
            repeats,
            BLOCK,
            num_warps=8,
        )
        return output

    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    out_view_stride = c_contiguous_stride(out_view_shape)

    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(output, out_view_shape, out_view_stride)
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask, other=0)
    out_offset = cumsum - repeats

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS_KUNLUNXIN REPEAT_INTERLEAVE_TENSOR")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"

    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()

    assert result_size >= 0, "repeats can not be negative"

    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    grid = (size,)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


@triton.jit
def repeat_interleave_self_tensor_bcast_kernel(
    inp_ptr,
    rep_ptr,
    cum_ptr,
    out_ptr,
    DIM_SIZE,
    IDX_TOTAL,
    INNER,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Load-once-store-k: one program handles one BLOCK-element chunk of one
    # input row (o, j) and writes its k = repeats[j] copies to the k
    # consecutive output rows. The input chunk is read exactly once instead of
    # once per output row (the per-output-row copy kernel re-reads it k times
    # on average, wasting read bandwidth for the common k > 1 case).
    j = ext.program_id(0)
    o = ext.program_id(1)
    c = ext.program_id(2)
    k = tl.load(rep_ptr + j).to(tl.int32)
    col = c * BLOCK + tl.arange(0, BLOCK)
    in_off = (o * DIM_SIZE + j).to(tl.int64) * INNER + col
    start = (tl.load(cum_ptr + j) - k).to(tl.int64) + o.to(tl.int64) * IDX_TOTAL
    if NEED_MASK:
        mask = col < INNER
        vals = tl.load(inp_ptr + in_off, mask=mask, other=0)
        for t in range(0, k):
            tl.store(out_ptr + (start + t) * INNER + col, vals, mask=mask)
    else:
        vals = tl.load(inp_ptr + in_off)
        for t in range(0, k):
            tl.store(out_ptr + (start + t) * INNER + col, vals)


@triton.jit
def repeat_interleave_self_tensor_gather_kernel(
    inp_ptr,
    index_ptr,
    out_ptr,
    IDX_TOTAL,
    DIM_SIZE,
    TOTAL_ROWS,
    BLOCK: tl.constexpr,
):
    # inner == 1 fast path: each output element is a scalar gather
    # out[o * IDX_TOTAL + j] = inp[o * DIM_SIZE + index[j]]
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ROWS
    r = offs
    j = r % IDX_TOTAL
    o = r // IDX_TOTAL
    src = tl.load(index_ptr + j, mask=mask, other=0)
    vals = tl.load(inp_ptr + (o * DIM_SIZE + src).to(tl.int64), mask=mask, other=0)
    tl.store(out_ptr + r.to(tl.int64), vals, mask=mask)


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_KUNLUNXIN REPEAT_INTERLEAVE_SELF_TENSOR")

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim, but got \
                repeats.size(0) = {} and input.size({}) = {}".format(
                repeats.size(0), dim, inp_shape[dim]
            )
        )

    cumsum = repeats.cumsum(axis=0)
    idx_total = int(cumsum[-1].item())

    outer = 1
    for s in inp_shape[:dim]:
        outer *= s
    inner = 1
    for s in inp_shape[dim + 1 :]:
        inner *= s

    inp = inp.contiguous()
    out_shape = inp_shape[:dim] + [idx_total] + inp_shape[dim + 1 :]
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    total_rows = outer * idx_total
    if total_rows == 0 or inner == 0:
        return out

    if inner == 1:
        # inner == 1 fast path: flat scalar gather, one BLOCK per program.
        indices = repeat_interleave_tensor(repeats)
        indices = indices.contiguous()
        BLOCK = 1024
        grid = (triton.cdiv(total_rows, BLOCK),)
        repeat_interleave_self_tensor_gather_kernel[grid](
            inp,
            indices,
            out,
            idx_total,
            inp_shape[dim],
            total_rows,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    # inner > 1: load-once-store-k. Each program reads one contiguous chunk of
    # one input row exactly once and stores its k = repeats[j] copies to the k
    # consecutive output rows (grid decomposed as (input row, outer, chunk)).
    block = min(triton.next_power_of_2(inner), 32768)
    chunks = triton.cdiv(inner, block)
    need_mask = (inner % block) != 0
    grid = (inp_shape[dim], outer, chunks)
    repeat_interleave_self_tensor_bcast_kernel[grid](
        inp,
        repeats,
        cumsum,
        out,
        inp_shape[dim],
        idx_total,
        inner,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=8 if block >= 4096 else 4,
        buffer_size_limit=8192 if block >= 8192 else 2048,
    )
    return out
