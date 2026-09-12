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

# Kunlunxin (XPU) special_expit(x) = 1 / (1 + exp(-x)).
#
# The generic flag_gems/ops/special_expit.py routes expit through
# tl_extra_shim.exp2 -> extern `_ZN3xpu5exp2fEf` (XPU software libdevice-style
# implementation, the same slow-extern family as the erfc/erf/lgamma fixes;
# baseline measures 79ms vs 0.21ms torch on (4096,4096) fp16, Gems Speedup
# ~0.0026x).  This override computes expit with the core triton math op
# tl.exp (math::ExpOp -> LLVM::Exp2Op, the fast native path already used by
# _kunlunxin/ops/sigmoid.py), so there is no extern call at all:
#   y = 1 / (1 + exp(-x))         computed in fp32, downcast at store.
# Accuracy vs a CPU fp64 reference (randn over all test shapes):
#   fp16 max abs err 2.44e-4  (0.22x of tolerance atol 1e-4 + rtol 1e-3*|ref|)
#   fp32 max abs err 5.96e-8  (0.0006x of atol 1e-4 + rtol 1.3e-6*|ref|)
#   bf16 max abs err 1.95e-3  (0.12x  of atol 1e-4 + rtol 0.016*|ref|)
# NaN/Inf semantics match torch: NaN propagates; +Inf -> 1.0; -Inf -> 0.0;
# +-0 -> 0.5.
#
# Tile buckets sweep-measured on XPU for this kernel (12 benchmark shapes x
# fp16/fp32/bf16, triton.testing.do_bench A/B in a single process, see
# harness/solution/performance/special_expit_perf.md); num_warps measures as a
# no-op on this backend, so buckets are chosen on tile width and dtype:
#   fp16: 16384/8 (>=1M), 4096/4 (262K), 8192/4 (65K), 2048/4 (below)
#   fp32: 65536/16 (>=4M), 16384/8 (1M-4M), 8192/4 (262K), 4096/4 (65K)  [vec]
#   bf16: 131072/32 (>=4M), 32768/8 (1M-4M), 16384/8 (262K), 8192/4 (65K) [vec]
# The "vec" variants set isCloseVectorization=True (skips the normalize/
# vectorize rewriter, the config used by the sibling _kunlunxin/ops/sigmoid.py
# kernel, which measures 20-35% faster than the default pass on bf16/fp32
# large tiles).  Unmasked runs when the shape divides the tile exactly
# (masked memory path on XPU costs ~2x); every benchmark shape divides its
# bucket, and non-dividing shapes fall back to the same tile with a mask.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _pick_block(n_elements, dtype):
    if n_elements >= 4_194_304:
        if dtype == torch.bfloat16:
            return 131072, 32, True
        if dtype == torch.float32:
            return 65536, 16, True
        return 16384, 8, False
    if n_elements >= 1_048_576:
        if dtype == torch.bfloat16:
            return 32768, 8, True
        return 16384, 8, False
    if n_elements >= 262_144:
        if dtype == torch.bfloat16:
            return 16384, 8, True
        if dtype == torch.float32:
            return 8192, 4, False
        return 4096, 4, False
    if n_elements >= 65_536:
        if dtype in (torch.float32, torch.bfloat16):
            return 4096, 4, dtype == torch.bfloat16
        return 8192, 4, False
    return 2048, 4, False


@triton.jit
def _expit_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expit_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, vec = _pick_block(n_elements, x.dtype)
    masked = n_elements % block_size != 0
    launch_kwargs = dict(
        num_warps=num_warps,
        unroll_num=8 if vec else 16,
        buffer_size_limit=4096 if vec else 8192,
    )
    if vec:
        launch_kwargs["isCloseVectorization"] = True
    else:
        launch_kwargs["isCloseMemoryAsync"] = False
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        _expit_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            **launch_kwargs,
        )
    else:
        grid = (n_elements // block_size,)
        _expit_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            **launch_kwargs,
        )


def special_expit(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_EXPIT")
    x = A.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out