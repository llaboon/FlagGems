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
import functools
import logging
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tle.raw fast path for the tensor-vs-scalar greater compare (P800 xpu3,
# cluster C payload in gt_raw.xpu). The compiler cannot vectorize a
# tensor-vs-scalar float compare (CoreTiling hands the scalar kernel 8
# elements/core/iteration, below the f32 vector width of 16, so `arith.cmpf`
# stays scalarized -- ~370us on 16M elements; TRITONXPU_VEC_REPORT shows the
# closure vetoed with keyState=Conflict). The hand-written payload streams the
# input once per core with pipelined GM2LM/LM2GM DMA and compares with the
# hardware vector-lt intrinsics (x > s evaluated as the ordered s < x), with
# the same memory footprint as the ATen reference. It also matches ATen's
# scalar-dtype promotion exactly. Mirrors the not_equal scalar payload
# (ne_raw.xpu / op-opti solution doc); see gt_raw.xpu for the full analysis.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1
# Must match CHUNK_BYTES in gt_raw.xpu (the chunk-grid partition contract).
_RAW_CHUNK_BYTES = 2048
# Below this element count the op is launch-bound and the bare pointwise
# scalar kernel (single launch, no extra host work) is fine; the payload wins
# from ~64K elements up (same crossover as the not_equal scalar payload).
_RAW_SMALL_SCALAR_LIMIT = 65536

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "gt_raw.xpu"))
    def gt_scalar_raw(in_, out, numel, esz, type_code, scalar_bits,
                      chunk_start, chunk_count):
        ...

    @triton.jit(do_not_specialize=["numel", "esz", "type_code", "scalar_bits",
                                   "chunk_count"])
    def gt_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits,
                             chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(gt_scalar_raw, (In, Out, numel, esz, type_code,
                                     scalar_bits, pid * chunk_count,
                                     chunk_count))


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


@functools.lru_cache(maxsize=1024)
def _scalar_bits(B, dtype):
    """The scalar promoted to `dtype`, as a sign-extended int32 bit pattern.

    Matches torch.greater's type promotion: the python float scalar is
    converted to the tensor's dtype and compared in that dtype. Cached: the
    dtype conversion costs a couple of microseconds on the host, which is
    directly visible on launch-bound small shapes.
    """
    if dtype == torch.float32:
        return int(torch.tensor(B, dtype=torch.float32).view(torch.int32).item())
    # fp16 / bf16: the payload only reads the low 16 bits.
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_greater_scalar(A, B, out=None):
    """greater(A, scalar) via the raw payload, or None when it does not apply.

    Only the contiguous case in the supported float dtypes is handled; when
    `out` is given it must be a contiguous bool tensor of A's shape (the
    payload fully overwrites it). Anything else falls back to the pointwise
    scalar kernel below.
    """
    if not _TLE_OK or not A.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(A.dtype)
    if type_code is None:
        return None
    M = A.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    if out is None:
        out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    elif not (
        out.dtype == torch.bool
        and out.is_contiguous()
        and out.shape == A.shape
    ):
        return None
    esz = A.element_size()
    s_bits = _scalar_bits(B, A.dtype)
    # partition by payload chunks (CHUNK_BYTES/esz elements each): every
    # program and core gets whole chunks so all GM2LM/LM2GM transfers are
    # CHUNK_BYTES-aligned in global memory.
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        gt_scalar_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(out), M, esz, type_code, s_bits, per)
    return out


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
    # Fast path: hand-written cluster-C payload (gt_raw.xpu). The compiler
    # scalarizes the tensor-vs-scalar compare (~370us on 16M elements, see the
    # header comment); the payload streams A once with per-core pipelined DMA
    # and the hardware vector-lt intrinsics at the same memory footprint as
    # ATen, and matches ATen's scalar-dtype promotion exactly.
    if A.numel() >= _RAW_SMALL_SCALAR_LIMIT:
        raw_out = _raw_greater_scalar(A, B)
        if raw_out is not None:
            return raw_out
    # NOTE: unlike the tensor path, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST. For tensor-vs-scalar
    # compare these fusion env vars make the compiler emit an fp16 compare that
    # trips `arith.cmpf requires all operands to have the same type` and blows the
    # uni_sram budget -> `out of resource: uni_sram` compile failure (fp16). The
    # sibling gt_scalar deliberately omits them for the same reason.
    res = greater_func_scalar(A, B)
    return res


# On the XPU backend a float compare whose second operand is a scalar is NOT
# vectorized (`tt.splat` of the scalar keeps `arith.cmpf` scalarized, no
# `triton_xpu.vcmpf`), so the tensor-vs-scalar kernel is ~4-5x slower than the
# tensor-vs-tensor kernel on large shapes (measured on [4096,4096] fp16: scalar
# 370us vs broadcast+tensor 84us). Below this element count the op is
# launch-bound and the single-launch scalar kernel is already faster than adding
# a full-size intermediate + second launch. The crossover was re-swept on the
# actual greater kernels: n=1M scalar 39.6us < bcast 59.3us, n=2M bcast 61.1us
# < scalar 63.7us, n>=4M bcast wins by 2-4x (fp16/fp32 consistent).
_SMALL_SCALAR_LIMIT = 2 * 1024 * 1024


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    # Fast path: hand-written cluster-C payload (gt_raw.xpu), writing straight
    # into `out` when it is a contiguous bool tensor of A's shape. The compiler
    # scalarizes the tensor-vs-scalar compare (see the header comment), and the
    # broadcast+tensor fallback below reads 2x the data; the payload streams A
    # once at the same memory footprint as ATen. Returns None only when the
    # payload does not apply (unsupported dtype/layout/out), in which case the
    # original paths below handle it.
    if A.numel() >= _RAW_SMALL_SCALAR_LIMIT:
        raw_out = _raw_greater_scalar(A, B, out)
        if raw_out is not None:
            return raw_out
    if A.numel() < _SMALL_SCALAR_LIMIT:
        # Small shapes: single launch, no intermediate tensor. Must NOT set
        # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST (fp16 tensor-vs-scalar
        # compare trips `arith.cmpf same-type` / uni_sram overflow -> compile
        # failure, see greater_scalar).
        if out is None:
            return greater_func_scalar(A, B)
        greater_func_scalar(A, B, out0=out)
        return out

    # Large shapes: the compiler scalarizes the tensor-vs-scalar compare, so
    # materialize the scalar as a contiguous full-size broadcast tensor
    # (torch.full_like rounds B to A's dtype, matching torch's scalar type
    # promotion) and route through the tuned tensor-tensor `greater_func` with
    # the fusion env vars, which lowers the compare to the vectorized fast path
    # (vcmpf). The given `out` is written in place (no result allocation).
    B_broadcast = torch.full_like(A, B)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        if out is None:
            res = greater_func(A, B_broadcast)
        else:
            greater_func(A, B_broadcast, out0=out)
            res = out
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return res
