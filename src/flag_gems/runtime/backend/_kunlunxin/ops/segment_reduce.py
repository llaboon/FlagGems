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

_BLOCK_SIZE = 1024
_UNIFORM_FAST_PATH_MIN_NUMEL = 1 << 16
_UNIFORM_KERNEL_MAX_SEGMENT_LENGTH = 1024
_UNIFORM_INNER_KERNEL_MAX_SEGMENT_LENGTH = 128
_UNIFORM_LENGTHS_CACHE = {}
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


def _get_block_size(device):
    return _BLOCK_SIZE


@triton.jit
def _mul_combine(a, b):
    return a * b


def _all_lengths_equal(lengths, value):
    # Cache the uniformity check keyed by tensor identity so repeated checks on the
    # same tensor (e.g. `_is_unit_lengths` then `_get_uniform_segment_length`) do not
    # trigger a host `.cpu()` synchronization every time. Holding a strong reference
    # to the tensor prevents `id()` reuse while the entry is alive; `_version` guards
    # against in-place mutation invalidating the cached verdict.
    cache_key = (lengths.device.type, id(lengths), value, getattr(lengths, "_version", None))
    is_equal = _UNIFORM_LENGTHS_CACHE.get(cache_key)
    if is_equal is None:
        is_equal = torch.all(lengths.detach().cpu() == value).item()
        if len(_UNIFORM_LENGTHS_CACHE) > 128:
            _UNIFORM_LENGTHS_CACHE.clear()
        _UNIFORM_LENGTHS_CACHE[cache_key] = is_equal
    return is_equal


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


def _get_uniform_segment_length(data, lengths, axis):
    if data.numel() < _UNIFORM_FAST_PATH_MIN_NUMEL:
        return None
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


def _is_unit_lengths(data, lengths, axis):
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return False
    if lengths.shape[-1] != data.shape[axis]:
        return False
    return _all_lengths_equal(lengths, 1)


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

    acc = tl.full((), INITIAL_VALUE, dtype=compute_dtype)
    has_nan = tl.full((), 0, dtype=tl.int32)

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
            has_nan += tl.where(is_nan, 1, 0)
            acc = tl.maximum(acc, tl.where(is_nan, float("-inf"), value))
        elif IS_MIN:
            is_nan = value != value
            has_nan += tl.where(is_nan, 1, 0)
            acc = tl.minimum(acc, tl.where(is_nan, float("inf"), value))
        pos += 1

    nan_value = tl.full((), float("nan"), dtype=compute_dtype)
    if IS_MEAN:
        acc_is_nan = acc != acc
        if not HAS_INITIAL:
            acc = tl.where(segment_length == 0, nan_value, acc)
        acc = tl.where((segment_length > 0) & ~acc_is_nan, acc / segment_length, acc)
    if IS_MAX or IS_MIN:
        acc = tl.where(has_nan > 0, nan_value, acc)

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
                if match:
                    counter += 1
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
                # XPU silently drops scalar masked stores, so materialize the mask
                # into the stored value instead (grad_input is pre-zeroed).
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
            nan_scalar = tl.full((), float("nan"), dtype=compute_dtype)
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
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    data_offsets = outer_idx * data_size_axis + dim_idx * segment_length + cols

    nan_scalar = tl.full((), float("nan"), dtype=compute_dtype)
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
        nan_mask = values != values
        has_nan = tl.max(tl.where(nan_mask, 1.0, 0.0), axis=1) > 0
        result = tl.max(values, axis=1)
        result = tl.where(has_nan, nan_scalar, result)
    elif IS_MIN:
        values = tl.load(data + data_offsets).to(compute_dtype)
        nan_mask = values != values
        has_nan = tl.max(tl.where(nan_mask, 1.0, 0.0), axis=1) > 0
        result = tl.min(values, axis=1)
        result = tl.where(has_nan, nan_scalar, result)

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
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    segment_start = dim_idx * segment_length
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

    has_nan = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int1)
    nan_value = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)

    for j in tl.static_range(BLOCK_SEG):
        data_offsets = base_offsets + j * inner_size
        if IS_SUM or IS_MEAN:
            values = tl.load(data + data_offsets).to(compute_dtype)
            acc += values
        elif IS_PROD:
            values = tl.load(data + data_offsets).to(compute_dtype)
            acc *= values
        elif IS_MAX:
            values = tl.load(data + data_offsets).to(compute_dtype)
            nan_mask = values != values
            has_nan |= nan_mask
            nan_value = tl.where(nan_mask, values, nan_value)
            acc = tl.maximum(acc, tl.where(nan_mask, float("-inf"), values))
        elif IS_MIN:
            values = tl.load(data + data_offsets).to(compute_dtype)
            nan_mask = values != values
            has_nan |= nan_mask
            nan_value = tl.where(nan_mask, values, nan_value)
            acc = tl.minimum(acc, tl.where(nan_mask, float("inf"), values))

    if IS_MEAN:
        acc = acc / segment_length
    if IS_MAX or IS_MIN:
        acc = tl.where(has_nan, nan_value, acc)

    output_offsets = rows * inner_size + k_offsets
    tl.store(output + output_offsets, acc)


@libentry()
@triton.jit
def _segment_reduce_uniform_sum_mean_backward_kernel(
    grad,
    grad_input,
    total_numel,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_MEAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Elementwise broadcast: every data element reads the single grad/output value of
    # its (outer, segment, inner) cell. This is a plain 1D grid-stride kernel with no
    # reduction, so it is safe on XPU and covers every uniform-length shape.
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel

    inner_idx = offsets % inner_size
    axis_idx = (offsets // inner_size) % data_size_axis
    outer_idx = offsets // (data_size_axis * inner_size)
    segment_idx = axis_idx // segment_length
    grad_offsets = (outer_idx * segment_count + segment_idx) * inner_size + inner_idx

    grad_value = tl.load(grad + grad_offsets, mask=mask, other=0.0)
    if IS_MEAN:
        grad_value = grad_value / segment_length
    tl.store(grad_input + offsets, grad_value, mask=mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_inner1_backward_kernel(
    grad,
    output,
    data,
    grad_input,
    total_rows,
    segment_count,
    segment_length,
    data_size_axis,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # inner_size == 1 backward. A 1D runtime while loop over the segment axis with
    # per-row vector accumulation -- no tl.sum/tl.reduce (miscompiled for the count
    # reduction) and no tl.static_range unroll (crashes with invalid PC at BLOCK_SEG
    # 64). This is the same pattern as the scalar fallback but vectorized over BLOCK_M
    # rows, and is reliable on this backend.
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < total_rows
    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    segment_start = dim_idx * segment_length
    base_offsets = outer_idx * data_size_axis + segment_start
    grad_value = tl.load(grad + rows, mask=mask, other=0.0).to(compute_dtype)
    output_value = tl.load(output + rows, mask=mask, other=0.0).to(compute_dtype)

    if IS_MAX or IS_MIN:
        counter = tl.zeros((BLOCK_M,), dtype=compute_dtype)
        pos = 0
        while pos < segment_length:
            data_offsets = base_offsets + pos
            values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
            match = (values != values) | (values == output_value)
            counter += tl.where(match, 1.0, 0.0)
            pos += 1
        store_value = tl.where(
            (counter >= 2.0) & (grad_value > 0),
            grad_value / counter,
            grad_value,
        )
        pos = 0
        while pos < segment_length:
            data_offsets = base_offsets + pos
            values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
            match = (values != values) | (values == output_value)
            tl.store(grad_input + data_offsets, tl.where(match, store_value, 0.0), mask=mask)
            pos += 1
    elif IS_PROD:
        zero_count = tl.zeros((BLOCK_M,), dtype=compute_dtype)
        nan_count = tl.zeros((BLOCK_M,), dtype=compute_dtype)
        product = tl.full((BLOCK_M,), INITIAL_PROD_VALUE, dtype=compute_dtype)
        pos = 0
        while pos < segment_length:
            data_offsets = base_offsets + pos
            values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
            is_nan = values != values
            is_zero = (values == 0) & ~is_nan
            nan_count += tl.where(is_nan, 1.0, 0.0)
            zero_count += tl.where(is_zero, 1.0, 0.0)
            product *= tl.where(is_nan | is_zero, 1.0, values)
            pos += 1

        zero_scalar = tl.zeros((BLOCK_M,), dtype=compute_dtype)
        nan_scalar = tl.full((BLOCK_M,), float("nan"), dtype=compute_dtype)
        normal_prefix = grad_value * output_value
        pos = 0
        while pos < segment_length:
            data_offsets = base_offsets + pos
            values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
            nan_mask = values != values
            zero_mask = (values == 0) & ~nan_mask
            normal_grad = normal_prefix / values
            zero_exclusive = tl.where(
                nan_count > 0, nan_scalar, tl.where(zero_count > 1, zero_scalar, product)
            )
            nan_exclusive = tl.where(
                nan_count > 1, nan_scalar, tl.where(zero_count > 0, zero_scalar, product)
            )
            exclusive = tl.where(nan_mask, nan_exclusive, zero_exclusive)
            grad_result = tl.where(
                nan_mask | zero_mask, grad_value * exclusive, normal_grad
            )
            tl.store(grad_input + data_offsets, grad_result, mask=mask)
            pos += 1


@libentry()
@triton.jit
def _segment_reduce_uniform_other_backward_kernel(
    grad,
    output,
    data,
    grad_input,
    total_rows,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SEG: tl.constexpr,
):
    # inner_size > 1: 2D tile [row, inner] with the segment axis fully unrolled at
    # compile time (tl.static_range). No tl.sum/tl.reduce over a runtime while, no
    # 3D tensors, no masked vector reduce -- the same pattern as the validated
    # forward static_range kernel.
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    segment_start = dim_idx * segment_length
    base_offsets = (
        outer_idx * data_size_axis * inner_size + segment_start * inner_size + k_offsets
    )
    output_offsets = rows * inner_size + k_offsets
    grad_value = tl.load(grad + output_offsets).to(compute_dtype)
    output_value = tl.load(output + output_offsets).to(compute_dtype)

    if IS_MAX or IS_MIN:
        counter = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)
        for j in tl.static_range(BLOCK_SEG):
            data_offsets = base_offsets + j * inner_size
            values = tl.load(data + data_offsets).to(compute_dtype)
            match = (values != values) | (values == output_value)
            counter += tl.where(match, 1.0, 0.0)
        store_value = tl.where(
            (counter >= 2.0) & (grad_value > 0),
            grad_value / counter,
            grad_value,
        )
        for j in tl.static_range(BLOCK_SEG):
            data_offsets = base_offsets + j * inner_size
            values = tl.load(data + data_offsets).to(compute_dtype)
            match = (values != values) | (values == output_value)
            tl.store(grad_input + data_offsets, tl.where(match, store_value, 0.0))
    elif IS_PROD:
        zero_count = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)
        nan_count = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)
        product = tl.full((BLOCK_M, BLOCK_K), INITIAL_PROD_VALUE, dtype=compute_dtype)
        for j in tl.static_range(BLOCK_SEG):
            data_offsets = base_offsets + j * inner_size
            values = tl.load(data + data_offsets).to(compute_dtype)
            is_nan = values != values
            is_zero = (values == 0) & ~is_nan
            nan_count += tl.where(is_nan, 1.0, 0.0)
            zero_count += tl.where(is_zero, 1.0, 0.0)
            product *= tl.where(is_nan | is_zero, 1.0, values)

        zero_scalar = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)
        nan_scalar = tl.full((BLOCK_M, BLOCK_K), float("nan"), dtype=compute_dtype)
        normal_prefix = grad_value * output_value
        for j in tl.static_range(BLOCK_SEG):
            data_offsets = base_offsets + j * inner_size
            values = tl.load(data + data_offsets).to(compute_dtype)
            nan_mask = values != values
            zero_mask = (values == 0) & ~nan_mask
            normal_grad = normal_prefix / values
            zero_exclusive = tl.where(
                nan_count > 0, nan_scalar, tl.where(zero_count > 1, zero_scalar, product)
            )
            nan_exclusive = tl.where(
                nan_count > 1, nan_scalar, tl.where(zero_count > 0, zero_scalar, product)
            )
            exclusive = tl.where(nan_mask, nan_exclusive, zero_exclusive)
            grad_result = tl.where(
                nan_mask | zero_mask, grad_value * exclusive, normal_grad
            )
            tl.store(grad_input + data_offsets, grad_result)


def _segment_reduce_uniform_backward(data, output, grad, reduce, lengths, axis):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None
    if segment_length > _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH:
        return None

    inner_size = _prod(data.shape[axis + 1 :])
    total_rows = _prod(lengths.shape)

    if reduce in ("sum", "mean"):
        grad_input = torch.empty_like(data, dtype=grad.dtype)
        if grad_input.numel() == 0:
            return grad_input
        block_size = _get_block_size(data.device)
        grid = (triton.cdiv(data.numel(), block_size),)
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_sum_mean_backward_kernel[grid](
                grad,
                grad_input,
                data.numel(),
                segment_count,
                segment_length,
                inner_size,
                data.shape[axis],
                reduce == "mean",
                BLOCK_SIZE=block_size,
            )
        return grad_input

    if reduce in ("max", "min"):
        grad_input = torch.zeros_like(data, dtype=grad.dtype)
    else:  # prod
        grad_input = torch.empty_like(data, dtype=grad.dtype)
    if grad_input.numel() == 0:
        return grad_input

    _, initial_prod_value = _make_initial("prod", None)

    # Runtime while-loop over the segment axis (no tl.sum/tl.reduce and no
    # tl.static_range unroll, both of which are miscompiled/crash on this backend for
    # the count reduction). inner_size == 1 vectorizes over rows only; inner_size > 1
    # uses a 2D tile over rows and the inner dimension (the inner dimension is vector
    # and only the small segment axis is unrolled).
    if segment_length > _UNIFORM_INNER_KERNEL_MAX_SEGMENT_LENGTH:
        return None
    if inner_size == 1:
        block_m = 32
        grid = (triton.cdiv(total_rows, block_m),)
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_inner1_backward_kernel[grid](
                grad,
                output,
                data,
                grad_input,
                total_rows,
                segment_count,
                segment_length,
                data.shape[axis],
                reduce == "max",
                reduce == "min",
                reduce == "prod",
                initial_prod_value,
                BLOCK_M=block_m,
            )
        return grad_input

    if total_rows % 4 != 0:
        return None
    if inner_size % 64 != 0:
        return None
    block_m = 4
    block_k = 64
    block_seg = segment_length
    grid = (total_rows // block_m, inner_size // block_k)
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_other_backward_kernel[grid](
            grad,
            output,
            data,
            grad_input,
            total_rows,
            segment_count,
            segment_length,
            inner_size,
            data.shape[axis],
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            initial_prod_value,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_SEG=block_seg,
        )
    return grad_input


def _segment_reduce_uniform_lengths(data, reduce, lengths, axis):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    output_shape = lengths.shape + data.shape[axis + 1 :]
    inner_size = _prod(data.shape[axis + 1 :])
    if segment_length > _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH:
        return None

    output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if output.numel() == 0:
        return output

    total_rows = _prod(lengths.shape)
    if inner_size == 1:
        # Single-shot 2D tile reduction (no mask, no runtime while loop). Requires
        # segment_length a power of two (tl.arange/tl.sum over BLOCK_N) and an exact
        # row tile so no row/col masking is needed.
        if segment_length & (segment_length - 1) != 0:
            return None
        if total_rows % 32 != 0:
            return None
        block_m = 32
        block_n = segment_length
        grid = (total_rows // block_m,)
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_inner1_forward_kernel[grid](
                data,
                output,
                total_rows,
                segment_count,
                segment_length,
                data.shape[axis],
                reduce == "sum",
                reduce == "mean",
                reduce == "max",
                reduce == "min",
                reduce == "prod",
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        return output

    # Compile-time unrolled 2D tile accumulation over the segment dimension (no
    # runtime scf.while, no masked load). Requires a small segment_length so the
    # static unroll does not blow up the IR, and exact row/inner tiles.
    if segment_length > _UNIFORM_INNER_KERNEL_MAX_SEGMENT_LENGTH:
        return None
    if total_rows % 4 != 0:
        return None
    if inner_size % 64 != 0:
        return None
    block_m = 4
    block_k = 64
    block_seg = segment_length
    grid = (total_rows // block_m, inner_size // block_k)
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_forward_kernel[grid](
            data,
            output,
            total_rows,
            segment_count,
            segment_length,
            inner_size,
            data.shape[axis],
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_SEG=block_seg,
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

        data_contig = data.contiguous()
        grad_contig = grad.contiguous()
        output_contig = output.contiguous()
        uniform_result = _segment_reduce_uniform_backward(
            data_contig, output_contig, grad_contig, reduce, lengths, axis
        )
        if uniform_result is not None:
            return uniform_result

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
