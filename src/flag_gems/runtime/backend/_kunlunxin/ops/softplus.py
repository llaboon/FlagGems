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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Kunlunxin/XPU performance override for aten::softplus / aten::softplus_backward.
#
# The previous override decorated the flat kernels with @libentry and always
# used a masked load/store (mask == offset < n_elements, constantly True for
# divisible shapes). On XPU two things hurt:
#   1. the per-call @libentry dispatch/keying overhead shows up directly in the
#      measured latency at small/medium shapes (same finding as sigmoid/silu
#      backward);
#   2. the always-true masked memory path serializes DMA vs the unmasked
#      contiguous-chunk path (~2-4x penalty, see log_sigmoid_forward).
#
# Following the established recipe (log_sigmoid_forward / acos / silu_backward)
# the kernels are plain @triton.jit, launched directly with an explicit
# num_warps tier and a NEED_MASK constexpr so that divisible shapes take the
# unmasked fast path. Kernel math is bit-identical to the previous version
# (fp32 intermediates, downcast at store, masked tail only when the shape does
# not divide the tile).

# (numel_upper_bound, BLOCK_SIZE, num_warps) following log_sigmoid_forward
# (same exp + log transcendental structure for the forward), plus a tiny tier
# so that sub-2048 tensors use a 1024-wide unmasked tile instead of a padded
# 2048 masked tile (wasted lanes dominate the launch-bound floor).
_TIERS = (
    (2048, 1024, 4),
    (16384, 2048, 4),
    (262144, 8192, 8),
    (None, 16384, 16),
)


def _pick_tier(numel):
    for hi, block, warps in _TIERS:
        if hi is None or numel <= hi:
            return block, warps
    return 16384, 16


@triton.jit(do_not_specialize=["n_elements", "beta", "threshold"])
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offset < n_elements
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offset).to(tl.float32)
    z = x * beta
    soft_z = tl.where(z > threshold, z, tl.log(1.0 + tl.exp(z)))
    out = soft_z / beta
    if NEED_MASK:
        tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n_elements", "beta", "threshold"])
def softplus_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offset < n_elements
        grad = tl.load(grad_ptr + offset, mask=mask, other=0.0)
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    else:
        grad = tl.load(grad_ptr + offset)
        x = tl.load(x_ptr + offset).to(tl.float32)
    z = x * beta
    derivative = tl.where(z > threshold, 1.0, tl.sigmoid(z))
    out = grad * derivative
    if NEED_MASK:
        tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty))


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS")
    x = self.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    block, warps = _pick_tier(n_elements)
    need_mask = (n_elements % block) != 0
    grid = (triton.cdiv(n_elements, block),)
    with torch_device_fn.device(x.device):
        softplus_kernel[grid](
            x,
            out,
            n_elements,
            beta,
            threshold,
            BLOCK_SIZE=block,
            NEED_MASK=need_mask,
            num_warps=warps,
        )
    return out


def softplus_backward(grad_output, self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS_BACKWARD")
    grad = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    x = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return out
    block, warps = _pick_tier(n_elements)
    need_mask = (n_elements % block) != 0
    grid = (triton.cdiv(n_elements, block),)
    with torch_device_fn.device(grad.device):
        softplus_backward_kernel[grid](
            grad,
            x,
            out,
            n_elements,
            beta,
            threshold,
            BLOCK_SIZE=block,
            NEED_MASK=need_mask,
            num_warps=warps,
        )
    return out
