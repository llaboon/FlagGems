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
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


# N above which the single-load 2D multirow tile no longer fits sram; fall back
# to the per-row online kernel.
MULTIROW_MAX_N = 8192


def _prev_pow2(x):
    x = max(1, int(x))
    return 1 << (x.bit_length() - 1)


def _multirow_tile_m(N):
    # Pack several rows per program so the [TILE_M, N] tile is one contiguous
    # block DMA. TILE_M is capped at 16: with masked tail rows a larger TILE_M
    # (e.g. 32) hits an XPU codegen bug that corrupts valid rows.
    return min(16, _prev_pow2(max(1, MULTIROW_MAX_N // N)))


# ------------------------  forward -------------------------------
# per-row program (grid=(M,)) + bounded TILE_N (heuristic "softmax_inner").
# Mirrors softmax_kernel_inner but final activation is  out = x - m - log(z).
@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        mask = n_offsets < N
        inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        log_z = tl.log(z)
        out = inp - m - log_z
        tl.store(output_ptr + offset, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last (partial) iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        log_z = tl.log(z)

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first store iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first").to(
                tl.float32
            )
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets, o)


# For small/medium N (<=MULTIROW_MAX_N) a per-row grid=(M,) launch is bandwidth
# starved on XPU (~4 GB/s). Pack TILE_M rows into a [TILE_M, N] constexpr-N tile
# so the whole tile is one contiguous block DMA and axis=1 reduce stays on-core.
@libentry()
@triton.jit
def log_softmax_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = m_offsets[:, None] < M
    inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(inp, 1)
    z = tl.sum(tl.exp(inp - m[:, None]), 1)
    out = inp - m[:, None] - tl.log(z)[:, None]
    tl.store(output_ptr + offsets, out, mask=mask)


# ------------------------  backward -------------------------------
# log_softmax backward:  scale = sum(out_grad over N); in_grad = out_grad - exp(out)*scale
#
# XPU dispatch (measured on P800): the old code sent all N<=8192 to the 2D
# [TILE_M, N] multirow tile. That tile does an axis=1 reduce that is pathological
# on XPU for medium N: as N grows TILE_M shrinks (=8192//N), the 2D reduce stops
# amortizing and gems latency explodes (N=4096 -> 14.7ms / sp 0.016, N=256 ->
# 3.0ms). Per-row 1D-reduce kernels (grid=(M,), base pointer pre-offset by pid*N
# so the offset tensor is pid-independent and stays block-DMA) are 8-17x faster
# for N>=512: N=4096 14.7->0.85ms, N=65536 43.7->19ms. But per-row is launch
# starved for tiny N (N=256 3.0->6.1ms), where the multirow row-packing still
# wins. So: N<=256 multirow, 256<N<=4096 single-pass per-row (out_grad cached in
# regs, one load), N>4096 two-pass per-row multi-tile (out_grad reloaded to avoid
# holding og+o simultaneously -> no reg spill on the wide tile).
BWD_MULTIROW_MAX_N = 4096
BWD_SINGLE_TILE_MAX_N = 4096
BWD_MT_TILE_N = 8192
# per-row two-pass tile width; measurable 17-20% faster than 8192 when the row
# length is an exact multiple of it (so there is no masked tail block).
BWD_MT_TILE_N_WIDE = 16384


# single-pass per-row: N fits one TILE_N tile, out_grad cached in registers.
@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        if NEED_MASK:
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)
        else:
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)


# two-pass per-row multi-tile: N>TILE_N, out_grad reloaded so the wide tile only
# ever holds one tensor at a time (avoids the reg spill that makes a single wide
# single-pass tile slow, e.g. N=8192 fp32 24.5ms single-pass vs 3.2ms two-pass).
@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow_mt(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N

        scale_acc = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale_acc += og
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale_acc += og
        scale = tl.sum(scale_acc, 0)

        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)


# N==1: reduction collapses to the element itself, so the whole op is a flat
# pointwise transform in_grad = og - exp(o) * og over the M flattened rows.
# BLOCK stays 256: at BLOCK=1024 the masked path of this kernel corrupts every
# 16th lane on XPU (multi-program shapes), while 256 is exact in every test.
@libentry()
@triton.jit
def log_softmax_backward_kernel_flat1(
    x_ptr,
    out_grad_ptr,
    in_grad_ptr,
    n_elem,
    BLOCK: tl.constexpr = 256,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    og = tl.load(out_grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ig = og - tl.exp(o) * og
    tl.store(in_grad_ptr + offs, ig, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig, mask=mask)
    else:
        og = tl.load(out_grad_ptr + offsets).to(tl.float32)
        o = tl.load(out_ptr + offsets).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig)


# tail rows of the multirow tile when M is not a multiple of TILE_M; launched
# with a single program after the unmasked full-tile launch.
@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    ROW_START,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = ROW_START + pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = m_offsets[:, None] < M
    og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(og, 1)
    ig = og - tl.exp(o) * scale[:, None]
    tl.store(in_grad_ptr + offsets, ig, mask=mask)


def _forward_launch(out, inp, M, N):
    if N <= MULTIROW_MAX_N:
        tile_m = _multirow_tile_m(N)
        grid = (triton.cdiv(M, tile_m), 1, 1)
        log_softmax_kernel_multirow[grid](
            out,
            inp,
            M,
            N,
            TILE_M=tile_m,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        grid = (M, 1, 1)
        log_softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            buffer_size_limit=2048,
            isCloseVectorization=True,
            is_use_mask_zero=True,
        )


def _backward_launch(output, grad_output, in_grad, M, N):
    if N == 1:
        # in_grad[i] = og[i] - exp(o[i]) * og[i]: pure flat pointwise.
        grid = (triton.cdiv(M, 256), 1, 1)
        log_softmax_backward_kernel_flat1[grid](
            output,
            grad_output,
            in_grad,
            M,
            buffer_size_limit=2048,
            num_warps=8,
        )
    elif N <= BWD_MULTIROW_MAX_N and (N & (N - 1)) == 0:
        # small/medium pow2 N: pack TILE_M rows per program into one [TILE_M, N]
        # contiguous block-DMA tile with the always-true row mask compiled away
        # (NEED_MASK=False); a masked single-program tail launch covers the
        # remaining M % TILE_M rows. non-pow2 N falls through to per-row.
        if N == 4096:
            # exception: a 4-row tile for N==4096 measures ~24% faster than the
            # 2-row default and is verified exact.
            tile_m = 4
        else:
            tile_m = min(16, _prev_pow2(max(1, 8192 // N)))
        nfull, tail = divmod(M, tile_m)
        if nfull:
            log_softmax_backward_kernel_multirow[(nfull, 1, 1)](
                output,
                grad_output,
                in_grad,
                M,
                N,
                TILE_M=tile_m,
                NEED_MASK=False,
                buffer_size_limit=2048,
                num_warps=8,
            )
        if tail:
            log_softmax_backward_kernel_multirow_tail[(1, 1, 1)](
                output,
                grad_output,
                in_grad,
                M,
                nfull * tile_m,
                N,
                TILE_M=tile_m,
                buffer_size_limit=2048,
                num_warps=8,
            )
    elif N <= BWD_SINGLE_TILE_MAX_N:
        # non-pow2 N <= 4096: masked single-pass per-row tile.
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=triton.next_power_of_2(N),
            NEED_MASK=True,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        # large N (N > 4096): per-row two-pass multi-tile. A 16384-wide tile is
        # 17-20% faster than 8192 for non-bf16; bf16 miscompiles at 16384-wide
        # so it always stays on the 8192-wide tile.
        tile_n = (
            BWD_MT_TILE_N if grad_output.dtype == torch.bfloat16 else BWD_MT_TILE_N_WIDE
        )
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow_mt[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=tile_n,
            buffer_size_limit=2048,
            num_warps=8,
        )


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            # reduction over an interior dim: transpose to make N contiguous,
            # merge (M, K) -> M' so the fast per-row inner kernel applies.
            inp_view = inp.view(M, N, K).transpose(1, 2).contiguous()
            inp_reshaped = inp_view.view(M * K, N)
            origin_dim = out.ndim
            if origin_dim == 3:
                m, n, k = out.shape
            elif origin_dim == 2:
                m, n = out.shape
            out_reshaped = torch.empty_like(inp_reshaped, dtype=dtype)

            _forward_launch(out_reshaped, inp_reshaped, M * K, N)
            if M == 1 and origin_dim == 2:
                out = out_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                out = out_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                out = out_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _forward_launch(out, inp, M, N)
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2).contiguous()
            out_view = output.view(M, N, K).transpose(1, 2).contiguous()
            out_grad_reshaped = out_grad_view.view(M * K, N)
            out_reshaped = out_view.view(M * K, N)
            in_grad_reshaped = torch.empty_like(out_reshaped, dtype=input_dtype)

            _backward_launch(
                out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N
            )
            origin_dim = output.ndim
            if origin_dim == 3:
                m, n, k = output.shape
            elif origin_dim == 2:
                m, n = output.shape
            if M == 1 and origin_dim == 2:
                in_grad = in_grad_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                in_grad = in_grad_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                in_grad = in_grad_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _backward_launch(output, grad_output, in_grad, M, N)
    return in_grad


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    if out.dtype != input_dtype:
        raise RuntimeError(
            f"_log_softmax_backward_data.out: expected out dtype {input_dtype}, got {out.dtype}"
        )

    K = output.numel() // M // N
    if K == 1 and out.is_contiguous():
        # Fast path: reduction over the last (contiguous) dim, the common
        # benchmark/tests case. Write directly into `out` so the .out variant
        # pays the same single-kernel cost as the functional variant (no extra
        # copy). The per-row kernels pre-offset the base pointers by pid*N and
        # store stride-1, so a contiguous [M, N] output is required here.
        grad_output_c = grad_output.contiguous()
        output_c = output.contiguous()
        with torch_device_fn.device(out.device):
            _backward_launch(output_c, grad_output_c, out, M, N)
        return out

    # Interior dim (K>1) or non-contiguous out: reuse the functional variant and
    # write back via the native strided-copy engine. gems overrides copy_/copy
    # but never _copy_from, so this reaches the vendor copy engine directly.
    in_grad = log_softmax_backward(grad_output, output, dim, input_dtype)
    torch.ops.aten._copy_from(in_grad, out, False)
    return out
