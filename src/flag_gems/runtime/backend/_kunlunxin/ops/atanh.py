# Kunlunxin (XPU) override of atanh / atanh_ / atanh_out.
#
# atanh was NOT overridden by kunlunxin, so torch.atanh fell to the generic
# KernelGen ops/atanh.py: a pointwise_dynamic (generic, no CodeGenConfig)
# kernel computing 0.5*log((1+x)/(1-x)) in fp32. Problems on XPU:
#   - performance: generic pointwise_dynamic has no tuned config_ (no
#     kunlunAutoGrid / prefer_1d_tile / unroll) -> discrete, per-shape-
#     recompiled access; measured 0.018x on large shapes (42ms for
#     [4096,4096] fp16, 168ms for [1024,65536] fp16, vs torch ~0.77/3.06ms).
#   - dispatch gap: ops/__init__.py registers ("atanh", atanh) but NOT
#     ("atanh_", atanh_), so torch.atanh_ under use_gems still fell through to
#     native xdnn atanh_, which raises
#     [NOT IMPLEMENTED] ... scalar type of ret : kbfloat16 is unsupported in
#     xdnn_pytorch_wrapper -> 6 bf16 functional failures (test_atanh_).
#     (_kunlunxin/__init__.py installs the extra-config entry, see there.)
#
# The numeric formula 0.5*log((1+x)/(1-x)) is the standard stable atanh:
#   - |x| -> 1: 1-x is exact (Sterbenz), (1+x)/(1-x) has ~2^-24 rel error
#     (verified: x=1-2^-24 -> 8.664340019 vs fp64 8.664339742, rel 3.2e-8);
#   - x -> 0: 1+-x rounds to 1 for |x|<2^-24 and the result becomes 0 instead
#     of x (absolute error <= |x| <= 6e-8, << test atol 1e-4);
#   - |x|>=1: (+/-)inf, |x|>1, NaN -> NaN; all match native torch semantics.
# Numeric probe vs fp64 on the full special-value set confirmed every value is
# within the test tolerance (rtol 1.3e-6 / atol 1e-4) -> no numerical fix
# needed, only the tuned config for the memory-bound win (same recipe as the
# sibling arcsinh.py / log1p.py: vec-closed + kunlunAutoGrid + unroll8).
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def atanh_func(x):
    # atanh(x) = 0.5 * log((1 + x) / (1 - x)), computed in fp32 for precision.
    # NOTE: no trailing .to(x.dtype) on purpose -- with an int64 input x stays
    # int64 in the kernel and NaN.to(int64) yields 2^63 garbage; the output
    # tensor store performs the final conversion instead (sibling exp/arcsinh
    # follow the same pattern).
    xf = x.to(tl.float32)
    return 0.5 * tl.log((1.0 + xf) / (1.0 - xf))


def atanh(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ATANH FORWARD")
    if out is None:
        return atanh_func(x)
    atanh_func(x, out0=out)
    return out


def atanh_(x):
    logger.debug("GEMS_KUNLUNXIN ATANH INPLACE")
    atanh_func(x, out0=x)
    return x


def atanh_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ATANH OUT")
    return atanh(x, out=out)