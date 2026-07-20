import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _nextafter_16bit_kernel(x_bits, y_bits, x, y):
    x_i32 = x_bits.to(tl.int32)
    y_i32 = y_bits.to(tl.int32)
    is_nan_x = x != x
    is_nan_y = y != y
    toward_pos_inf = y > x
    step = tl.where((x_i32 < 0) ^ toward_pos_inf, 1, -1)
    result = x_i32 + step
    result = tl.where((x_i32 == 0) & ~toward_pos_inf, -32768, result)
    result = tl.where((x_i32 == -32768) & toward_pos_inf, 1, result)
    result = tl.where(x == y, y_i32, result)
    result = tl.where(is_nan_x, x_i32, result)
    result = tl.where(is_nan_y, y_i32, result)
    return result.to(tl.int16)


def nextafter_(input, other):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER_")
    if input.dtype in (torch.float16, torch.bfloat16):
        if input.dtype != other.dtype:
            raise RuntimeError(
                "Found dtype %s but expected %s" % (other.dtype, input.dtype)
            )
        if other.shape != input.shape:
            other = other.expand(input.shape)
        kernel = _nextafter_16bit_kernel.instantiate(input.ndim)
        kernel(
            input.view(torch.int16),
            other.view(torch.int16),
            input,
            other,
            out0=input.view(torch.int16),
        )
        return input

    from flag_gems.ops.nextafter_ import nextafter_ as generic_nextafter_

    return generic_nextafter_(input, other)
