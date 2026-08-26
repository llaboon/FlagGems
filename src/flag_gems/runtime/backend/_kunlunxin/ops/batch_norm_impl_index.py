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
from torch import Tensor

from .batch_norm import batch_norm, batch_norm_backward

logger = logging.getLogger(__name__)


def batch_norm_impl_index(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    cudnn_enabled=True,
):
    # aten::batch_norm (functional) is a CompositeImplicitAutograd op that
    # internally redispatches to aten::_batch_norm_impl_index, so every
    # batch_norm-family entry point (F.batch_norm, native_batch_norm_legit,
    # instance_norm, ...) funnels here. The generic impl_index kernel's 2-D
    # strided pointer arithmetic fails XPU tt.addptr verification ("result
    # type matches ptr type", _batch_norm_impl_index.py:172) and surfaces as
    # OutOfResources/uni_sram on both training and eval paths. Delegate to
    # the kunlunxin 3-stage batch_norm implementation instead.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_IMPL_INDEX")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_invstd, reserve, 0


def batch_norm_impl_index_backward(
    impl_index,
    input,
    grad_out,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    train=False,
    eps=1e-05,
    output_mask=None,
    reservedSpace=None,
):
    # aten::_batch_norm_impl_index_backward(int impl_index, Tensor input,
    # Tensor grad_output, ... bool[3] output_mask, Tensor reservedSpace):
    # impl_index comes FIRST; keep the formal order in schema order.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_IMPL_INDEX_BACKWARD")
    if output_mask is None:
        output_mask = [True, True, True]
    return batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,  # PyTorch's save_var is in fact inv_std
        train,
        eps,
        output_mask,
    )


def batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    # aten::_batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias,
    # Tensor? running_mean, Tensor? running_var, float momentum, float eps)
    # -> (Tensor, Tensor, Tensor, Tensor). Eval-only: normalize with the
    # running stats, never update them. CPU returns empty fp32 save tensors
    # and an empty uint8 reserve; the generic kernel trips the XPU compiler
    # (RuntimeError: PassManager::run failed) on every shape.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_NO_UPDATE")
    output, _, _ = batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )
    empty_f32 = torch.empty((0,), dtype=torch.float32, device=input.device)
    empty_u8 = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), empty_f32, empty_f32, empty_u8


def batch_norm_with_update_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    # aten::_batch_norm_with_update_functional is called directly by
    # torch.ops.aten; the generic kernel shares the impl_index addptr
    # failure. Delegate to the kunlunxin 3-stage path, whose training
    # branch folds the running-stat updates in place (combine kernel),
    # so the mutated running_mean/running_var tensors are the new stats.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_WITH_UPDATE_FUNCTIONAL")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        True,
        momentum,
        eps,
    )
    # aten schema returns SIX tensors: (output, save_mean, save_var,
    # reserve, running_mean_out, running_var_out).
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return (
        output.view_as(input),
        save_mean,
        save_invstd,
        reserve,
        running_mean,
        running_var,
    )


def miopen_batch_norm_backward(
    input,
    grad_output,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_var,
    epsilon,
):
    # aten::miopen_batch_norm_backward mirrors the cudnn variant. Its
    # save_var argument holds the saved inverse standard deviation (rstd),
    # which is exactly what batch_norm_backward expects as save_invstd.
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_BACKWARD")
    input_grad, weight_grad, bias_grad = batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )
    return input_grad, weight_grad, bias_grad
