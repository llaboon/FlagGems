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

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _native_batch_norm_legit(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    # aten::_native_batch_norm_legit mutates running stats in place and
    # returns (output, save_mean, save_invstd). The kunlunxin batch_norm
    # training branch folds the running-stat update into the combine
    # kernel, so the in-place mutation is covered.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT")
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    if not training:
        # aten CPU returns EMPTY save tensors in eval mode.
        empty = torch.empty(0, dtype=torch.float32, device=input.device)
        return output, empty, empty
    return output, save_mean, save_invstd


def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    if not training:
        raise RuntimeError("Expected training to be true, but got false.")
    channels = input.shape[1]
    running_mean = torch.zeros(channels, dtype=input.dtype, device=input.device)
    running_var = torch.ones(channels, dtype=input.dtype, device=input.device)
    return batch_norm(
        input, weight, bias, running_mean, running_var, True, momentum, eps
    )


def _copy_outputs(result, out, save_mean, save_invstd):
    result_out, result_mean, result_invstd = result
    out.resize_as_(result_out).copy_(result_out)
    save_mean.resize_as_(result_mean).copy_(result_mean)
    save_invstd.resize_as_(result_invstd).copy_(result_invstd)
    return out, save_mean, save_invstd


def _native_batch_norm_legit_out(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_OUT")
    result = _native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_no_stats_out(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    result = _native_batch_norm_legit_no_stats(
        input, weight, bias, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_functional(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    # Functional variant: the caller's running stats must NOT be mutated;
    # updated stats are returned as new tensors. Clone before delegating
    # so the in-place update inside batch_norm lands on the clones.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")
    rm = running_mean.clone()
    rv = running_var.clone()
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, rm, rv, training, momentum, eps
    )
    return output, save_mean, save_invstd, rm, rv


def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
):
    # Eval-only variant: normalize with the running stats, no update.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )
    return output, save_mean, save_invstd
