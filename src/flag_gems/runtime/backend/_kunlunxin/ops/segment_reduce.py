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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_REDUCES = ("sum", "mean", "max", "min", "prod")
_SUPPORTED_DATA_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _prod(shape):
    return math.prod(shape) if shape else 1


def _wrap_axis(axis, ndim):
    if ndim == 0:
        raise IndexError(
            "segment_reduce(): input tensor must have at least one dimension."
        )
    if axis < -ndim or axis >= ndim:
        raise IndexError(
            f"segment_reduce(): axis {axis} is out of bounds for tensor of dimension {ndim}."
        )
    return axis % ndim


def _check_reduce_and_dtype(data, reduce):
    if reduce not in _SUPPORTED_REDUCES:
        raise RuntimeError(
            "segment_reduce(): reduce must be one of 'sum', 'mean', 'max', 'min', or 'prod'."
        )
    if data.dtype not in _SUPPORTED_DATA_DTYPES:
        raise NotImplementedError(f'"segment_reduce" not implemented for {data.dtype}.')


def _check_index_tensor(data, index_tensor, name, axis):
    if index_tensor.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise NotImplementedError(f"segment_reduce(): {name} must be int32 or int64.")
    if index_tensor.device != data.device:
        raise RuntimeError(
            f"segment_reduce(): Expected data and {name} on the same device."
        )
    if data.dim() < index_tensor.dim():
        raise RuntimeError(
            f"segment_reduce(): Expected data.dim() >= {name}.dim(), got "
            f"{data.dim()} and {index_tensor.dim()}."
        )
    if axis != index_tensor.dim() - 1:
        raise RuntimeError(
            f"segment_reduce(): Expected axis to be the last dimension of {name} "
            f"but got {axis}."
        )


def _validate_lengths(data, lengths, axis, unsafe):
    _check_index_tensor(data, lengths, "lengths", axis)
    if unsafe:
        return
    lengths_cpu = lengths.detach().cpu()
    if torch.any(lengths_cpu < 0).item():
        raise RuntimeError("lengths contains negative value!")
    valid_lengths = torch.all(lengths_cpu.sum(dim=-1) == data.size(axis)).item()
    if not valid_lengths:
        raise RuntimeError(
            "segment_reduce(): Expected all rows of lengths along axis to sum to "
            "data.size(lengths.dim()-1) when !unsafe."
        )


def _make_initial(reduce, initial):
    if initial is not None:
        return True, initial
    if reduce == "max":
        return False, float("-inf")
    if reduce == "min":
        return False, float("inf")
    if reduce == "prod":
        return False, 1.0
    return False, 0.0


@libentry()
@triton.jit
def _lengths_to_offsets_kernel(
    lengths,
    offsets,
    outer_count,
    segment_count,
):
    pid = tle.program_id(0)
    acc = tl.full((), 0, dtype=tl.int64)
    base_lengths = pid * segment_count
    base_offsets = pid * (segment_count + 1)
    tl.store(offsets + base_offsets, acc)

    idx = 0
    while idx < segment_count:
        length = tl.load(lengths + base_lengths + idx)
        acc += length
        tl.store(offsets + base_offsets + idx + 1, acc)
        idx += 1


@libentry()
@triton.jit
def _segment_reduce_forward_kernel(
    data,
    offsets,
    output,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    INITIAL_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # NOTE: On the XPU backend a vectorized masked reduction
    # (tl.arange + tl.load(mask=...) + tl.sum inside a while loop over a
    # runtime segment bound) crashes `TritonXPUMask` with a `uni_sram`
    # OutOfResources error.  We therefore reduce each segment with a scalar
    # accumulation loop (no tl.arange / tl.sum / masked load), which compiles
    # and is numerically equivalent.
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    inner_idx = pid % inner_size
    row_idx = pid // inner_size
    dim_idx = row_idx % segment_count
    outer_idx = row_idx // segment_count

    offsets_base = outer_idx * (segment_count + 1) + dim_idx
    segment_start = tl.load(offsets + offsets_base)
    segment_end = tl.load(offsets + offsets_base + 1)
    segment_length = segment_end - segment_start

    acc = tl.full((), INITIAL_VALUE, dtype=compute_dtype)
    has_nan = tl.zeros((), dtype=tl.int1)
    nan_value = tl.zeros((), dtype=compute_dtype)

    pos = segment_start
    while pos < segment_end:
        data_offset = (
            outer_idx * data_size_axis * inner_size + pos * inner_size + inner_idx
        )
        value = tl.load(data + data_offset).to(compute_dtype)
        if IS_SUM or IS_MEAN:
            acc += value
        elif IS_PROD:
            acc *= value
        elif IS_MAX:
            is_nan = value != value
            has_nan = has_nan | is_nan
            nan_value = tl.where(is_nan, value, nan_value)
            acc = tl.maximum(acc, tl.where(is_nan, float("-inf"), value))
        elif IS_MIN:
            is_nan = value != value
            has_nan = has_nan | is_nan
            nan_value = tl.where(is_nan, value, nan_value)
            acc = tl.minimum(acc, tl.where(is_nan, float("inf"), value))
        pos += 1

    if IS_MEAN:
        acc_is_nan = acc != acc
        nan_value_mean = acc / acc
        if not HAS_INITIAL:
            acc = tl.where(segment_length == 0, nan_value_mean, acc)
        acc = tl.where((segment_length > 0) & ~acc_is_nan, acc / segment_length, acc)
    if IS_MAX or IS_MIN:
        acc = tl.where(has_nan, nan_value, acc)

    tl.store(output + pid, acc)


@libentry()
@triton.jit
def _segment_reduce_backward_kernel(
    grad,
    output,
    data,
    offsets,
    grad_input,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    inner_idx = pid % inner_size
    row_idx = pid // inner_size
    dim_idx = row_idx % segment_count
    outer_idx = row_idx // segment_count

    offsets_base = outer_idx * (segment_count + 1) + dim_idx
    segment_start = tl.load(offsets + offsets_base)
    segment_end = tl.load(offsets + offsets_base + 1)
    segment_length = segment_end - segment_start

    if segment_length > 0:
        grad_value = tl.load(grad + pid).to(compute_dtype)
        output_value = tl.load(output + pid).to(compute_dtype)

        if IS_SUM or IS_MEAN:
            if IS_MEAN:
                grad_value = grad_value / segment_length
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                tl.store(grad_input + data_offset, grad_value)
                pos += 1
        elif IS_MAX or IS_MIN:
            counter = tl.full((), 0, dtype=tl.int64)
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                value = tl.load(data + data_offset).to(compute_dtype)
                match = (value != value) | (value == output_value)
                counter += match.to(tl.int64)
                pos += 1

            store_value = tl.where(
                (counter >= 2) & (grad_value > 0),
                grad_value / counter.to(compute_dtype),
                grad_value,
            )
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                value = tl.load(data + data_offset).to(compute_dtype)
                match = (value != value) | (value == output_value)
                # NOTE: scalar masked store (mask=scalar) is silently ignored on
                # XPU, so materialize the mask into the stored value instead.
                store_val = tl.where(match, store_value, 0.0)
                tl.store(grad_input + data_offset, store_val)
                pos += 1
        elif IS_PROD:
            zero_count = tl.full((), 0, dtype=tl.int64)
            nan_count = tl.full((), 0, dtype=tl.int64)
            product = tl.full((), INITIAL_PROD_VALUE, dtype=compute_dtype)
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                value = tl.load(data + data_offset).to(compute_dtype)
                if value != value:
                    nan_count += 1
                elif value == 0:
                    zero_count += 1
                else:
                    product *= value
                pos += 1

            zero_scalar = tl.full((), 0.0, dtype=compute_dtype)
            nan_scalar = zero_scalar / zero_scalar
            normal_prefix = grad_value * output_value
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                value = tl.load(data + data_offset).to(compute_dtype)
                nan_mask = value != value
                zero_mask = (value == 0) & ~nan_mask
                normal_grad = normal_prefix / value
                zero_exclusive = tl.where(
                    nan_count > 0,
                    nan_scalar,
                    tl.where(zero_count > 1, zero_scalar, product),
                )
                nan_exclusive = tl.where(
                    nan_count > 1,
                    nan_scalar,
                    tl.where(zero_count > 0, zero_scalar, product),
                )
                exclusive = tl.where(nan_mask, nan_exclusive, zero_exclusive)
                grad_result = tl.where(
                    nan_mask | zero_mask,
                    grad_value * exclusive,
                    normal_grad,
                )
                tl.store(grad_input + data_offset, grad_result)
                pos += 1


@triton.jit
def _mul_combine(a, b):
    return a * b


@libentry()
@triton.jit
def _segment_reduce_uniform_inner1_forward_kernel(
    data,
    output,
    total_rows,
    segment_count,
    segment_length,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Uniform, power-of-two segment_length, inner_size == 1.  No masking is
    # needed (divisibility is guaranteed by the caller), so this is a pure
    # unmasked vectorized reduction -- avoiding the XPU `TritonXPUMask`
    # `uni_sram` crash triggered by masked load + tl.sum inside a runtime
    # while loop.
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    data_offsets = outer_idx * data_size_axis + dim_idx * segment_length + cols

    if IS_SUM or IS_MEAN:
        values = tl.load(data + data_offsets).to(compute_dtype)
        result = tl.sum(values, axis=1)
        if IS_MEAN:
            result = result / segment_length
    elif IS_PROD:
        values = tl.load(data + data_offsets).to(compute_dtype)
        result = tl.reduce(values, axis=1, combine_fn=_mul_combine)
    elif IS_MAX:
        values = tl.load(data + data_offsets).to(compute_dtype)
        result = tl.max(values, axis=1)
        # tl.max does not propagate NaN on XPU (it ignores NaN lanes), but
        # torch's segment_reduce("max") does.  Detect NaN lanes and re-insert
        # NaN so the result matches the reference semantics.
        has_nan = tl.max((values != values).to(compute_dtype), axis=1) > 0
        result = tl.where(has_nan, float("nan"), result)
    elif IS_MIN:
        values = tl.load(data + data_offsets).to(compute_dtype)
        result = tl.min(values, axis=1)
        has_nan = tl.max((values != values).to(compute_dtype), axis=1) > 0
        result = tl.where(has_nan, float("nan"), result)

    output_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(output + output_offsets, result)


@libentry()
@triton.jit
def _segment_reduce_uniform_forward_kernel(
    data,
    output,
    total_rows,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SEG: tl.constexpr,
):
    # Uniform segment_length, inner_size > 1.  A 2D [BLOCK_M, BLOCK_K] tile is
    # reduced element-wise over the segment dimension.  The segment loop is
    # fully unrolled at compile time via `tl.static_range(BLOCK_SEG)` (BLOCK_SEG
    # == segment_length), producing straight-line vectorized loads with no
    # runtime `scf.while` loop.  This avoids both the XPU `TritonXPUMask`
    # `uni_sram` crash (masked load + tl.sum inside a runtime loop) and the
    # `arith.addi` type-mismatch crash triggered by a 2D-tile scalar-offset
    # accumulation inside a runtime while loop.
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    segment_start = dim_idx * BLOCK_SEG
    base_offsets = (
        outer_idx * data_size_axis * inner_size + segment_start * inner_size + k_offsets
    )

    if IS_MAX:
        acc = tl.full((BLOCK_M, BLOCK_K), float("-inf"), dtype=compute_dtype)
    elif IS_MIN:
        acc = tl.full((BLOCK_M, BLOCK_K), float("inf"), dtype=compute_dtype)
    elif IS_PROD:
        acc = tl.full((BLOCK_M, BLOCK_K), 1.0, dtype=compute_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)

    for j in tl.static_range(BLOCK_SEG):
        values = tl.load(data + base_offsets + j * inner_size).to(compute_dtype)
        if IS_SUM or IS_MEAN:
            acc += values
        elif IS_PROD:
            acc *= values
        elif IS_MAX:
            acc = tl.maximum(acc, values)
        elif IS_MIN:
            acc = tl.minimum(acc, values)

    if IS_MEAN:
        acc = acc / BLOCK_SEG

    output_offsets = rows * inner_size + k_offsets
    tl.store(output + output_offsets, acc)


def _segment_reduce_uniform_lengths(data, reduce, lengths, axis):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    output_shape = lengths.shape + data.shape[axis + 1 :]
    inner_size = _prod(data.shape[axis + 1 :])
    output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if output.numel() == 0:
        return output

    total_rows = _prod(lengths.shape)
    data_size_axis = data.shape[axis]

    if inner_size == 1:
        # Single-shot reduction over segment_length.  Require power-of-two and
        # a bounded segment_length so BLOCK_N exactly covers the segment (no
        # tail mask, and tl.sum stays within the safe XPU block bound).
        if segment_length & (segment_length - 1) != 0 or segment_length > 1024:
            return None
        block_m = 32
        block_n = segment_length
        if total_rows % block_m != 0:
            return None
        grid = (triton.cdiv(total_rows, block_m),)
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_inner1_forward_kernel[grid](
                data,
                output,
                total_rows,
                segment_count,
                segment_length,
                data_size_axis,
                reduce == "sum",
                reduce == "mean",
                reduce == "max",
                reduce == "min",
                reduce == "prod",
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        return output

    # Fully-unrolled segment loop: cap segment_length so the compile-time
    # unroll does not explode the IR for pathological segment lengths.
    if segment_length > 128:
        return None
    block_m = 4
    block_k = 64
    if total_rows % block_m != 0 or inner_size % block_k != 0:
        return None
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(inner_size, block_k))
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_forward_kernel[grid](
            data,
            output,
            total_rows,
            segment_count,
            segment_length,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_SEG=segment_length,
        )
    return output


def _lengths_to_offsets(lengths):
    segment_count = lengths.shape[-1]
    offsets_shape = lengths.shape[:-1] + (segment_count + 1,)
    offsets = torch.empty(offsets_shape, dtype=lengths.dtype, device=lengths.device)
    outer_count = _prod(lengths.shape[:-1])
    if offsets.numel() > 0:
        with torch_device_fn.device(lengths.device):
            _lengths_to_offsets_kernel[(outer_count,)](
                lengths,
                offsets,
                outer_count,
                segment_count,
            )
    return offsets


def _prepare_common(data, reduce, lengths, offsets, indices, axis, unsafe):
    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    if indices is not None:
        raise RuntimeError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    if offsets is not None:
        _check_index_tensor(data, offsets, "offsets", axis)
        offsets_contig = offsets.contiguous()
        segment_count = offsets_contig.shape[-1] - 1
        output_shape = (
            offsets_contig.shape[:-1] + (segment_count,) + data.shape[axis + 1 :]
        )
        return axis, offsets_contig, output_shape, True

    if lengths is None:
        raise RuntimeError(
            "segment_reduce(): Either lengths or offsets must be defined."
        )

    _validate_lengths(data, lengths, axis, unsafe)
    lengths_contig = lengths.contiguous()
    offsets_contig = _lengths_to_offsets(lengths_contig)
    output_shape = lengths_contig.shape + data.shape[axis + 1 :]
    return axis, offsets_contig, output_shape, False


_UNIFORM_LENGTHS_CACHE = {}


def _all_lengths_equal(lengths, value):
    # Cache the (device-side) uniformity check keyed on the tensor's identity
    # (id) plus a strong reference to the tensor.  Holding the strong reference
    # guarantees the id is never recycled to a different tensor while the entry
    # is live, and the `is` identity check + `_version` guard make the cache
    # immune to stale results from data-pointer reuse or in-place mutation.
    # This avoids a `.cpu()` synchronization on every call, which dominates the
    # unit-length fast path for tiny shapes.
    key = (id(lengths), value)
    entry = _UNIFORM_LENGTHS_CACHE.get(key)
    if (
        entry is not None
        and entry[0] is lengths
        and entry[1] == getattr(lengths, "_version", None)
    ):
        return entry[2]
    is_equal = torch.all(lengths.detach().cpu() == value).item()
    if len(_UNIFORM_LENGTHS_CACHE) > 128:
        _UNIFORM_LENGTHS_CACHE.clear()
    _UNIFORM_LENGTHS_CACHE[key] = (
        lengths,
        getattr(lengths, "_version", None),
        is_equal,
    )
    return is_equal


def _is_unit_lengths(data, lengths, axis):
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return False
    if lengths.shape[-1] != data.shape[axis]:
        return False
    return _all_lengths_equal(lengths, 1)


def _get_uniform_segment_length(data, lengths, axis):
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return None
    segment_count = lengths.shape[-1]
    if segment_count <= 0:
        return None
    data_size_axis = data.shape[axis]
    if data_size_axis % segment_count != 0:
        return None
    segment_length = data_size_axis // segment_count
    if segment_length <= 0:
        return None
    if _all_lengths_equal(lengths, segment_length):
        return segment_length
    return None


def segment_reduce(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
):
    logger.debug("GEMS_KUNLUNXIN SEGMENT_REDUCE")
    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    if indices is not None:
        raise RuntimeError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    if initial is None and lengths is not None and offsets is None:
        _check_index_tensor(data, lengths, "lengths", axis)
        if _is_unit_lengths(data, lengths, axis):
            return data.contiguous()

        data_contig = data.contiguous()
        uniform_result = _segment_reduce_uniform_lengths(
            data_contig, reduce, lengths, axis
        )
        if uniform_result is not None:
            return uniform_result

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, indices, axis, unsafe
    )

    data_contig = data.contiguous()
    output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if output.numel() == 0:
        return output

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    data_size_axis = data_contig.shape[axis]
    has_initial, initial_value = _make_initial(reduce, initial)
    grid = (output.numel(),)

    with torch_device_fn.device(data.device):
        _segment_reduce_forward_kernel[grid](
            data_contig,
            offsets_contig,
            output,
            segment_count,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            has_initial,
            initial_value,
            BLOCK_SIZE=1024,
        )
    return output


def segment_reduce_out(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SEGMENT_REDUCE_OUT")
    result = segment_reduce(
        data,
        reduce,
        lengths=lengths,
        indices=indices,
        offsets=offsets,
        axis=axis,
        unsafe=unsafe,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _segment_reduce_backward(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
):
    logger.debug("GEMS_KUNLUNXIN _SEGMENT_REDUCE_BACKWARD")
    if (
        initial is None
        and lengths is not None
        and offsets is None
        and reduce in _SUPPORTED_REDUCES
    ):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        if _is_unit_lengths(data, lengths, axis):
            return grad.contiguous()

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, None, axis, True
    )
    data_contig = data.contiguous()
    grad_contig = grad.contiguous()
    output_contig = output.contiguous()
    grad_input = torch.zeros(data_contig.shape, dtype=grad.dtype, device=grad.device)

    if output_contig.numel() == 0:
        return grad_input

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    data_size_axis = data_contig.shape[axis]
    _, initial_prod_value = _make_initial("prod", initial)
    grid = (output_contig.numel(),)

    with torch_device_fn.device(data.device):
        _segment_reduce_backward_kernel[grid](
            grad_contig,
            output_contig,
            data_contig,
            offsets_contig,
            grad_input,
            segment_count,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            initial_prod_value,
            BLOCK_SIZE=1024,
        )
    return grad_input


def _segment_reduce_backward_out(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
    out,
):
    logger.debug("GEMS_KUNLUNXIN _SEGMENT_REDUCE_BACKWARD_OUT")
    result = _segment_reduce_backward(
        grad,
        output,
        data,
        reduce,
        lengths=lengths,
        offsets=offsets,
        axis=axis,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out
