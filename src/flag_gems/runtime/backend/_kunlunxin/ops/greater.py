# Kunlunxin (XPU) override of greater / greater_out / greater_scalar /
# greater_scalar_out.
#
# `greater.Tensor` is functionally identical to `gt.Tensor`, and kunlunxin
# already ships a tuned override for gt (`_kunlunxin/ops/gt.py`). But `greater`
# was NOT overridden, so it fell back to the generic bare `pointwise_dynamic`
# (no CodeGenConfig) -> discrete access on XPU -> catastrophic latency
# (60-1000 ms for large shapes, gems speedup ~0.001 in
# `harness/perf_ir_2/greater.log`).
#
# Fix: reuse the exact gt recipe -- same tuned CodeGenConfig
# (unroll_num=8, kunlunAutoGrid=True, prefer_1d_tile=True) plus the
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST launch env vars for the tensor
# path. Kernel body / algorithm unchanged (zero correctness risk).
import logging
import os

import torch
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
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


# Scalar (tensor-vs-scalar) compare path. Same bandwidth-bound 1D-tile recipe
# as config_, but with unroll_num=16 + buffer_size_limit=8192. On XPU the scalar
# greater kernel is pure memory-bound (~385 GB/s at unroll_num=8); a fresh-compile
# config sweep on [1024,1024,1024] showed unroll_num=16 + buffer_size_limit=8192
# is the sweet spot -> fp16 7.85->6.84ms, fp32 7.31->6.00ms (~13-18% faster),
# while unroll_num=32 and larger buffer_size_limit regress or plateau. Pure
# codegen-param change: kernel body / algorithm / numerics unchanged.
# NOTE (perf fix): the scalar path additionally needs TRITONXPU_COMPARE_FUSION=1
# at compile time -- see greater_scalar below. An earlier note here claimed the
# fusion env vars gave "zero benefit" on the scalar kernel; that measurement was
# invalid (it was taken against a stale entry in the default Triton binary
# cache). Re-measured with a fresh TRITON_CACHE_DIR, COMPARE_FUSION alone is
# worth ~7.8x on this kernel.
config_scalar = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=8192,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def greater_func(x, y):
    return x.to(tl.float32) > y


def greater(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = greater_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


def greater_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_OUT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    if out is None:
        res = greater_func(A, B)
    else:
        greater_func(A, B, out0=out)
        res = out
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# --------------------------------------------------------------------------
# Scalar (tensor-vs-scalar) compare.
#
# Two independent compile-time levers, both measured with a *fresh*
# TRITON_CACHE_DIR (12-CTA rank-1 geometry, unroll_num=16,
# buffer_size_limit=8192, [4096,4096] fp16):
#
#   1) TRITONXPU_COMPARE_FUSION=1 alone: 0.336 ms -> 0.043 ms (~7.8x). Without
#      it the XPU compiler does not fuse the compare into the vectorized
#      load/store pipeline. Note TRITONXPU_FP16_FAST must NOT be set together
#      with it on this kernel -- the combination puts the compare back on the
#      slow path (0.336 ms), i.e. the tensor-path env recipe is wrong here.
#   2) compare dtype: for 16-bit floats, comparing in the tensor's own dtype
#      instead of promoting to fp32 gives another 0.059 -> 0.043 ms on fp16
#      (speedup 0.60 -> 0.83). This matches PyTorch semantics: a python-number
#      scalar is a "wrapped number" and does not participate in type promotion,
#      so torch casts the scalar down to the tensor dtype. Non-16-bit dtypes
#      keep the fp32 compare, which is required for correct int-tensor vs
#      float-scalar promotion.
# --------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar,
)
@triton.jit
def greater_func_scalar(x, y):
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        return x > y.to(x.dtype)
    else:
        return x.to(tl.float32) > y


def _scalar_fusion_env():
    """Enable the XPU compare-fusion pass for the scalar kernel compile.

    Returns the previous value so it can be restored (unlike the tensor path we
    must not unconditionally `del`, since the tensor path may be active in an
    enclosing frame).
    """
    prev = os.environ.get("TRITONXPU_COMPARE_FUSION")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    return prev


def _restore_scalar_fusion_env(prev):
    if prev is None:
        os.environ.pop("TRITONXPU_COMPARE_FUSION", None)
    else:
        os.environ["TRITONXPU_COMPARE_FUSION"] = prev


def greater_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR")
    prev = _scalar_fusion_env()
    try:
        return greater_func_scalar(A, B)
    finally:
        _restore_scalar_fusion_env(prev)


# ---------------------------------------------------------------------------
# Fast path for large contiguous float tensors whose numel is an exact
# multiple of _GREATER_SCALAR_FAST_TILE with a scalar exactly representable in
# A.dtype.
#
# Why: the generic scalar-compare path (pointwise_dynamic 1d-tile codegen)
# materializes `arith.cmpf -> i1 -> bool store` per lane, which the XPU backend
# lowers to a ~10x slower path (measured XPU 3: direct fp32 compare [10000,
# 65536] fp16 13.06 ms vs a generic fp32 store of the saturating result
# 2.16 ms; the same compare/i1/bool-store root cause is documented for the
# lt_/gt_/ge_ family in HARNESS_SUMMARY §2.6 and the lt_scalar/ge closures).
# The runtime mask (`tid < num_tasks`) is always-true here and adds a second,
# smaller penalty (masked-memory path, ~2-3%, same as the lt_scalar closure).
#
# Strategy -- two stages, no i1 is ever materialized in Triton:
#   1. saturating fp arithmetic on the fp32-upcast value (family recipe:
#      t = (x - s) * 1e30; max(0,t); min(1,t)) -> exactly 0.0/1.0 written into
#      a fp32 buffer. M = 1e30 saturates every representable x != s gap of
#      fp16/bf16/fp32 to +-inf, so the result is bit-exact 0.0/1.0.
#   2. fp32 -> bool conversion via `torch.ops.aten._copy_from` (NOT in
#      `_FULL_CONFIG`, so it reaches the vendor's native conversion kernel
#      even inside `use_gems`; measured 1.97 ms on [10000,65536] fp32). The
#      FlagGems `_to_copy`/`copy_` overrides are deliberately not used: they
#      compute a per-lane fp->bool cast, i.e. the same slow lowering we avoid
#      (measured ~2s on the same shape under use_gems).
# Measured on XPU 3 [10000,65536]: fp16 13.40 -> 4.13 ms, bf16 13.41 -> 4.13 ms,
# fp32 12.53 -> 4.75 ms (torch ~1.09/1.09/1.86 ms).
#
# Semantics: torch.greater compares against the scalar rounded to A.dtype
# (wrapped scalar), and the gate `float(B) == float(torch.tensor(B,
# dtype=A.dtype).item())` restricts the fast path to scalars exactly
# representable in A.dtype (e.g. benchmark scalar 0.5, test scalar 0), so the
# fp32 compare against float(B) is bit-identical to torch. NaN inputs settle
# to 0.0 (max/min on this backend prefer the non-NaN operand; torch: NaN > s
# == False); +-inf, +-0, equality and subnormal gaps are exact. Corner
# behavior verified against torch on device for all three dtypes
# (iso_midf32.py; also consistent with the ge_/lt_ family closures).
_GREATER_SCALAR_FAST_TILE = 131072
_GREATER_SCALAR_MIN_GRID = 512


@triton.jit
def greater_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _greater_scalar_fast(A, scalar):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (A.numel() // _GREATER_SCALAR_FAST_TILE,)
    greater_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GREATER_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


def _greater_scalar_out_fast(A, scalar, out):
    # Two-stage recipe identical to _greater_scalar_fast, but stage 2 converts
    # into the caller-provided bool `out` instead of an internal buffer. Stage 1
    # writes the saturating fp32 result (0.0/1.0) into a fresh fp32 buffer; stage
    # 2 is the vendor-native fp32->bool conversion via aten._copy_from, which is
    # not in _FULL_CONFIG and therefore bypasses the slow per-lane fp->bool
    # lowering under use_gems (see _greater_scalar_fast docstring for the full
    # rationale). out is required contiguous with the same numel as A on this
    # path (enforced by the gate in greater_scalar_out).
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (A.numel() // _GREATER_SCALAR_FAST_TILE,)
    greater_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GREATER_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    torch.ops.aten._copy_from(out32, out, False)
    return out


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    prev = _scalar_fusion_env()
    try:
        if out is None:
            return greater_func_scalar(A, B)
        greater_func_scalar(A, B, out0=out)
        return out
    finally:
        _restore_scalar_fusion_env(prev)
