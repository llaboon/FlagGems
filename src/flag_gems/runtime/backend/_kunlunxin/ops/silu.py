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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import libentry, tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


# silu_backward uses a dedicated bounded-tile kernel on XPU. The previous
# pointwise_dynamic implementation (tile = next_pow2(numel/12), up to 2M-wide
# per CTA) combined with `div_rn` (IEEE round-to-nearest division, ~2.9x slower
# than plain `/` on XPU) and `isCloseVectorization/unroll_num` left large shapes
# at ~0.51 gems speedup. This custom kernel uses a bounded BLOCK (<= 65536,
# more CTAs for the compute-heavy exp), plain division (within torch tolerance),
# and `buffer_size_limit=4096` with vectorization open: measured [4096,4096]
# fp16 speedup 0.51 -> ~1.02, fp32 0.51 -> ~0.92, bf16 0.63 -> ~0.79.
_SILU_BW_MAX_BLOCK = 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def silu_backward_kernel_xpu(
    x_ptr, dy_ptr, out_ptr, n_elements, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_elements
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + tid, mask=mask).to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(out_ptr + tid, dx.to(x_ptr.type.element_ty), mask=mask)


# silu_backward tiny fast path (contiguous fp16/fp32/bf16, numel <= 2048):
# a flat 1D masked/unmasked kernel that skips the pointwise_dynamic wrapper.
# Measurement on XPU 5 (2026-08-19, official 12-shape matrix, do_bench A/B):
# at numel <= 2048 the pointwise codegen (kunlunAutoGrid=False) pays a fixed
# ~8us wrapper/grid overhead per call (e.g. [1024,1] fp16 15.5us vs 7.1us flat);
# at numel > 2048 the tuned pointwise config_ is strictly faster than every
# flat/NEED_MASK tier (B2048..B32768 x w4..16) and every CodeGenConfig variant
# (unroll 8/16/32 x buffer 4096/8192/16384 x tile 256/512/1024 x autogrid),
# so only the tiny window uses the flat kernel. Math bit-identical to
# silu_backward_kernel (fp32 staging, div_rn, downcast at store).
_TINY_MAX_NUMEL = 2048
_TINY_BLOCK = 2048
_TINY_WARPS = 4


@triton.jit
def silu_backward_tiny_kernel(
    g_ptr, x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        dy = tl.load(g_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
        dy = tl.load(g_ptr + offs)
    x_fp32 = x.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    if NEED_MASK:
        tl.store(out_ptr + offs, dx.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, dx.to(x.dtype))


def _silu_backward_tiny(grad_output, self):
    numel = grad_output.numel()
    out = torch.empty_like(self)
    if numel == 0:
        return out
    if numel == _TINY_BLOCK:
        silu_backward_tiny_kernel[(1,)](
            grad_output,
            self,
            out,
            numel,
            BLOCK=_TINY_BLOCK,
            NEED_MASK=False,
            num_warps=_TINY_WARPS,
        )
    else:
        silu_backward_tiny_kernel[(1,)](
            grad_output,
            self,
            out,
            numel,
            BLOCK=_TINY_BLOCK,
            NEED_MASK=True,
            num_warps=_TINY_WARPS,
        )
    return out


def silu(self):
    logger.debug("GEMS_KUNLUNXIN SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN SILU_BACKWARD")
    if (
        grad_output.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and grad_output.numel() <= _TINY_MAX_NUMEL
    ):
        return _silu_backward_tiny(grad_output, self)
    x = self if self.is_contiguous() else self.contiguous()
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    grad_input = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements > 0:
        block = min(triton.next_power_of_2(n_elements), _SILU_BW_MAX_BLOCK)
        grid = (triton.cdiv(n_elements, block), 1, 1)
        silu_backward_kernel_xpu[grid](
            x,
            dy,
            grad_input,
            n_elements,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    if grad_input.shape != self.shape or grad_input.stride() != self.stride():
        grad_input = grad_input.reshape(self.shape).as_strided(
            self.size(), self.stride()
        )
    return grad_input


def silu_(A):
    logger.debug("GEMS_KUNLUNXIN SILU_")
    out = silu_forward(A, out0=A)
    return out
