import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


PI = 3.141592653589793
PI_2 = 1.5707963267948966


# ------------------------------------------------------------
# asin polynomial on small interval
# valid roughly on [0, 0.62]
# ------------------------------------------------------------

@triton.jit
def asin_poly_small(x):
    x2 = x * x
    # Horner 法则计算泰勒展开到 x^15
    # 避免产生大量中间 IR 节点，同时精度满足 fp32 要求
    y = 0.01396484  # C15
    y = 0.01735276 + x2 * y # C13
    y = 0.02237216 + x2 * y # C11
    y = 0.03038194 + x2 * y # C9
    y = 0.04464286 + x2 * y # C7
    y = 0.07500000 + x2 * y # C5
    y = 0.16666667 + x2 * y # C3
    y = 1.0         + x2 * y # C1
    y = x * y
    return y


# ------------------------------------------------------------
# asin implementation
# ------------------------------------------------------------

@triton.jit
def asin_custom(x):
    PI_2 = 1.5707963267948966
    x_fp32 = x.to(tl.float32)

    sign = x_fp32 < 0.0

    ax = tl.abs(x_fp32)

    # --------------------------------------------------------
    # range reduction
    #
    # asin(x) =
    # pi/2 - 2*asin(sqrt((1-x)/2))
    #
    # for x > 0.62
    # --------------------------------------------------------

    large = ax > 0.62

    reduced = tl.sqrt((1.0 - ax) * 0.5)
    
    poly_input = tl.where(large, reduced, ax)

    y = asin_poly_small(poly_input)

    y = tl.where(
        large,
        PI_2 - 2.0 * y,
        y,
    )

    y = tl.where(sign, -y, y)

    return y


# ------------------------------------------------------------
# acos implementation
# ------------------------------------------------------------

@triton.jit
def acos_custom(x):
    PI_2 = 1.5707963267948966
    asin_x = asin_custom(x)

    return PI_2 - asin_x


# ------------------------------------------------------------
# kernel
# ------------------------------------------------------------
from flag_gems.utils import tl_extra_shim
_acos = tl_extra_shim.acos
@pointwise_dynamic(
    promotion_methods=[(0, "INT_TO_FLOAT")]
)
@triton.jit()
def acos_kernel(x):
    return _acos(x.to(tl.float32))
    return acos_custom(x)

def acos(x):

    logger.debug("GEMS_KUNLUNXIN ACOS_CUSTOM")

    y = acos_kernel(x)
    
    return y
