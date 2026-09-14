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
import struct

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

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


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


# Tuned scalar-path variants (config_): the untuned kernels above are
# launch-optimal for small tensors, so they are kept for
# numel < DIV_SCALAR_CFG_THRESHOLD; larger tensors use the Kunlunxin tuned
# CodeGenConfig. Measured on XPU (scalar path, same do_bench):
#   (4096,4096) fp32 221us -> 142us, bf16 224us -> 149us, fp16 206us -> 168us;
#   (2^28,) fp32 3317us -> 2053us, fp16 3089us -> 2292us, bf16 3369us -> 2171us;
#   (10000,256) fp32 41us -> 30us. Small tensors are untouched (launch floor).
DIV_SCALAR_CFG_THRESHOLD = 1 << 20


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def true_div_func_tensor_scalar_cfg(x, y):
    return x / y


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def true_div_func_scalar_tensor_cfg(x, y):
    return x / y


# fp32-only unroll-16 variant of the tuned scalar kernel: measured on XPU for
# the in-place scalar path (out0=A, same do_bench median, 3 interleaved
# rounds) that unroll_num=16 is strictly faster than unroll 8 for fp32 across
# the whole shape range ((4096,4096) 141.9->138.4us, (10000,256) 30.3->29.6us,
# (2^28,) 2043->1998us, (10000,65536) 4949->4842us; ~2.2%), while fp16/bf16
# are flat or slightly worse with unroll 16 (fp16 (2^28,) 2292->2341us), so
# the unroll-16 kernel is dispatched only for fp32 in true_divide_.
CFG_UNROLL16 = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=CFG_UNROLL16,
)
@triton.jit
def true_div_func_tensor_scalar_cfg16(x, y):
    return x / y


# Tensor/tensor unroll-16 variant (CFG_UNROLL16): measured on XPU with
# isolated processes (official do_bench warmup=1000ms/rep=100ms median) that
# unroll 16 is strictly faster than unroll 8 (config_) for fp16/fp32 at
# numel >= 4M, while bf16 regresses at all sizes and numel < 4M explodes:
#   fp16 (4096,4096) 0.996->1.114x  (1024,65536) 1.023->1.148x
#        (64,64,65536) 1.042->1.170x (64,64,4096) 0.993->1.114x
#   fp32 (4096,4096) 0.703->0.731x  (1024,65536) 0.717->0.761x
#        (1024,4096) 0.67->0.72x    (64,64,4096) ~0.70->0.73x
#   bf16 (4096,4096) 0.960->0.852x  (1024,65536) 0.986->0.865x (regress)
#   fp32 (1024,256) 1.02->0.045x    (1024,16) 1.02->0.185x (explode)
# -> dispatch unroll16 only for fp16/fp32 tensor/tensor at numel >= 4M;
# bf16 and smaller tensors keep config_ (true_div_func).
DIV_TENSOR_U16_MIN_NUMEL = 1 << 22  # 4M


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=CFG_UNROLL16)
@triton.jit
def true_div_func_u16(x, y):
    return x / y


def div_complex_real(ar, ai, br, bi):
    # Smith's method: divide by the larger denominator component to avoid
    # intermediate overflow/underflow (mirrors the common op complex kernel).
    # Computed in fp32: fp16/bf16 components would lose precision and the
    # fp16 division path is less robust in the XPU backend.
    arf = ar.to(tl.float32)
    aif = ai.to(tl.float32)
    brf = br.to(tl.float32)
    bif = bi.to(tl.float32)
    abs_br = tl.abs(brf)
    abs_bi = tl.abs(bif)
    use_br = abs_br >= abs_bi

    # When |br| >= |bi|: ratio = bi/br, denom = br + bi*ratio
    ratio1 = tl.where(brf == 0, 0.0, bif / brf)
    denom1 = brf + bif * ratio1
    real1 = (arf + aif * ratio1) / denom1
    imag1 = (aif - arf * ratio1) / denom1

    # When |bi| > |br|: ratio = br/bi, denom = bi + br*ratio
    ratio2 = tl.where(bif == 0, 0.0, brf / bif)
    denom2 = bif + brf * ratio2
    real2 = (arf * ratio2 + aif) / denom2
    imag2 = (aif * ratio2 - arf) / denom2

    return tl.where(use_br, real1, real2)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    promotion_methods=[(0, 1, 2, 3, "INT_TO_FLOAT")],
)
@triton.jit
def div_complex_imag(ar, ai, br, bi):
    arf = ar.to(tl.float32)
    aif = ai.to(tl.float32)
    brf = br.to(tl.float32)
    bif = bi.to(tl.float32)
    abs_br = tl.abs(brf)
    abs_bi = tl.abs(bif)
    use_br = abs_br >= abs_bi

    ratio1 = tl.where(brf == 0, 0.0, bif / brf)
    denom1 = brf + bif * ratio1
    imag1 = (aif - arf * ratio1) / denom1

    ratio2 = tl.where(bif == 0, 0.0, brf / bif)
    denom2 = bif + brf * ratio2
    imag2 = (aif * ratio2 - arf) / denom2

    return tl.where(use_br, imag1, imag2)


def _true_divide_complex(A, B):
    # Kunlunxin pointwise codegen cannot lower complex pointers
    # (canonicalize_ptr_dtype KeyError), so compute Smith's method on
    # real/imag components with two plain pointwise kernels and reassemble.
    if not A.is_complex():
        # real ÷ complex: promote A to B's precision, imag = 0
        a = torch.stack((A.to(B.dtype), torch.zeros_like(A, dtype=B.dtype)), dim=-1)
    else:
        a = torch.view_as_real(A.resolve_conj().resolve_neg())
    if isinstance(B, torch.Tensor):
        if B.is_complex():
            b = torch.view_as_real(B.resolve_conj().resolve_neg())
        else:
            b = torch.stack(
                (B.to(a.dtype), torch.zeros_like(B, dtype=a.dtype)), dim=-1
            )
    else:
        if isinstance(B, complex):
            b = torch.tensor(
                [B.real, B.imag], dtype=a.dtype, device=a.device
            ).unsqueeze(0).expand(*a.shape[:-1], 2)
        else:
            b = torch.stack(
                (
                    torch.full(a.shape[:-1], B, dtype=a.dtype, device=a.device),
                    torch.zeros(a.shape[:-1], dtype=a.dtype, device=a.device),
                ),
                dim=-1,
            )
    a = a.contiguous()
    b = b.contiguous()
    ar, ai = a.select(-1, 0), a.select(-1, 1)
    br, bi = b.select(-1, 0), b.select(-1, 1)
    real = div_complex_real(ar, ai, br, bi)
    imag = div_complex_imag(ar, ai, br, bi)
    output = torch.stack((real, imag), dim=-1)
    return torch.view_as_complex(output)


def divide(A, B):
    # Vendor entry for aten.divide.Tensor. SpecOpRegistrar replaces the
    # flag_gems global by function name, so a function named `divide` must be
    # exported here; otherwise divide.Tensor keeps dispatching to the generic
    # flag_gems.ops.divide.divide -> generic true_div_func (untuned codegen
    # config), which runs ~300x slower than the tuned kunlunxin kernel.
    #
    # tests/test_divide.py pins the legacy dispatch log contract ("GEMS DIVIDE"
    # via logger "flag_gems.ops.divide"); emit the same message through that
    # logger so the contract holds while the computation runs on the tuned
    # kunlunxin kernel below.
    logging.getLogger("flag_gems.ops.divide").debug("GEMS DIVIDE")
    logger.debug("GEMS_KUNLUNXIN DIVIDE")
    return true_divide(A, B)


def true_divide_tensor_(A, B):
    # Vendor entry for aten.true_divide_.Tensor. Tensor.true_divide_ dispatches
    # here even for scalar others, so without a function named
    # `true_divide_tensor_` exported from this package, SpecOpRegistrar keeps
    # the generic flag_gems.ops.true_divide_.true_divide_tensor_ (which imports
    # the generic true_divide_ by value), and the untuned generic kernel runs
    # ~300x slower than the tuned kunlunxin kernel.
    #
    # tests/test_true_divide.py pins the legacy dispatch log contract
    # ("GEMS TRUE_DIVIDE_" via logger "flag_gems.ops.true_divide_") to prove the
    # call is intercepted by a flag_gems override rather than native aten. Emit
    # the same message through that logger so the contract holds while the
    # actual computation runs on the tuned kunlunxin kernel below.
    logging.getLogger("flag_gems.ops.true_divide_").debug("GEMS TRUE_DIVIDE_")
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR_")
    return true_divide_(A, B)


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and A.is_complex():
        return _true_divide_complex(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and B.is_complex():
        return _true_divide_complex(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B)
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.is_complex():
            # The pointwise code generator has no complex scalar dtype mapping.
            # Divide interleaved real/imag lanes with the existing Triton kernel.
            return torch.view_as_complex(
                true_div_func_tensor_scalar(torch.view_as_real(A), B)
            )
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_tensor_scalar_cfg(A, B)
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_scalar_tensor_cfg(A, B)
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_tensor(A, B):
    """Canonical Tensor overload of true_divide (explicit aten true_divide.Tensor).

    The generic flag_gems.ops.true_divide.true_divide_tensor routes through the
    generic flag_gems.ops.div.true_divide, whose pointwise kernel lacks the
    Kunlunxin tuned CodeGenConfig (measured ~330x slower on XPU for
    (4096,4096) fp32: 69ms vs 205us). Exporting this vendor implementation lets
    SpecOpRegistrar swap it in so torch.true_divide(tensor, tensor) and
    aten::true_divide.Tensor use the same fast tuned kernel as div/div_.
    """
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR")
    # keep the dispatch-contract log line expected by tests/test_true_divide.py
    # (caplog on logger "flag_gems.ops.true_divide", same pattern as special_erf)
    logging.getLogger("flag_gems.ops.true_divide").debug("GEMS TRUE_DIVIDE")
    return true_divide(A, B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_OUT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B, out0=out)
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_tensor_scalar_cfg(A, B, out0=out)
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_scalar_tensor_cfg(A, B, out0=out)
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        # Both scalar
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B, out0=A)
        return true_div_func(A, B, out0=A)
    else:
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            if A.dtype == torch.float32:
                return true_div_func_tensor_scalar_cfg16(A, B, out0=A)
            return true_div_func_tensor_scalar_cfg(A, B, out0=A)
        return true_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _trunc_q(q):
    # Truncate a fp32 quotient toward zero without the slow `xpu_trunc`
    # extern call (measured ~2.3x slower on XPU). fp32 -> int32 cast
    # (fptosi, round toward zero) is exact for |q| < 2^23; for |q| >= 2^23
    # the fp32 value is already integral (23-bit mantissa), so keep it.
    return tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)


@triton.jit
def _floor_q(q):
    # floor() of a fp32 quotient (fast path for the common case where the
    # fp32 quotient is well away from an integral boundary). int32 cast is
    # RTZ (truncation): for q >= 0 floor == trunc; for q < 0 and q not
    # integral subtract 1. |q| >= 2^23 fp32 values are already integral.
    t = tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)
    return tl.where((q < 0) & (q != t), t - 1.0, t)


@triton.jit
def _floor_div_fp32(x, y):
    # floor(x/y) matching the torch-CPU / numpy reference (npy_divmodf)
    # semantics bit-for-bit, without the slow XPU fp64 path:
    #   mod = fmodf(x, y)                    (fp32, single-rounded fma)
    #   if (mod != 0 && isless(b,0) != isless(mod,0)) { mod += b; }
    #   div = (x - mod) / y
    #   fd = floor(div); if (div - fd > 0.5) fd += 1  (snap tie cases)
    #   if (div == 0) fd = copysign(0, x/y)
    # This is what the CPU reference (used by --ref cpu) computes; the old
    # fp64-division implementation was emulated on XPU (~8-10x slower) and
    # also differed from the CPU reference near integral boundaries.
    q = div_rn(x, y)
    t = tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)
    mod0 = tl.fma(t, -y, x)
    adj = (mod0 != 0.0) & ((y < 0.0) != (mod0 < 0.0))
    div = div_rn(x - mod0, y)  # numpy: div computed with original mod
    div = tl.where(adj, div - 1.0, div)
    fd = _floor_q(div)
    fd = tl.where(div - fd > 0.5, fd + 1.0, fd)
    fd = tl.where(
        div == 0.0,
        tl.where((x < 0.0) != (y < 0.0), -0.0, 0.0),
        fd,
    )
    # division by zero: numpy returns a / b directly (signed inf)
    return tl.where(y == 0.0, q, fd)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def trunc_div_func(x, y):
    return _trunc_q(div_rn(x, y))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return _trunc_q(div_rn(x, tl.cast(y, x.dtype)))


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return _trunc_q(div_rn(tl.cast(x, y.dtype), y))


# Integer truncation division: Triton's // on integers is C-style (truncates toward zero)
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y):
    return x // y


def trunc_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE_")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    # TODO: request Triton to add an integer remainder builtin
    # The semantic of Triton floordiv differs from Pytorch/Numpy
    # Triton floordiv equates to
    #     (x - np.fmod(x, y)) / y
    # whereas Pytorch floordiv is
    #     (x - np.remainder(x, y)) y
    # The results show a one off difference when
    #     C1) x and y have opposite signs
    # and C2) x is not multiples of y.
    # Apart from the above, there's an erroneous case x // 0 returns -1
    # whereas in Pytorch x // 0 returns -1 if x >=0 and -2 if x < 0
    # but this special case is coalesced into the c1 and c2 check so
    # there's extra handling.
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


# floor_divide must be consistent with python/numpy/torch: floor(x/y) on the
# fp32 device quotient, implemented with the fast int-cast floor (fp64 is
# emulated on XPU and ~8-10x slower; the old fp64 path also differed from
# torch by 1 ulp near integral boundaries since torch divides in fp32).
@triton.jit
def _float_floordiv_corrected(x, y):
    # x, y are already fp32. The old fp64-division implementation was
    # emulated on XPU (~8-10x slower) and also differed from torch by 1 ulp
    # near integral boundaries. New: fp32 device division + exact-remainder
    # floor (matches CPU reference semantics bit-wise).
    return _floor_div_fp32(x, y)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def floor_div_func_corrected(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def floor_div_func_corrected_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def floor_div_lowp_tensor_scalar_func(x, y):
    # fp16/bf16 scalar path: promote to fp32 and divide in fp32 (torch does
    # the same; there is no native fp16/bf16 division), then floor exactly
    # with the same exact-remainder semantics as the fp32 path.
    y = tl.full(x.shape, y, x.dtype)
    return _floor_div_fp32(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def floor_div_func_corrected_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


def _as_bfloat16_scalar(value):
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent != 0x7F800000:
        bits += 0x7FFF + ((bits >> 16) & 1)
    elif mantissa:
        bits |= 0x00400000
    bits &= 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def floor_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func_corrected(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype in (torch.float16, torch.bfloat16):
            if A.dtype == torch.bfloat16:
                B = _as_bfloat16_scalar(B)
            return floor_div_lowp_tensor_scalar_func(A, B)
        return floor_div_func_corrected_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_corrected_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func_corrected(A, B, out0=A)
    else:
        if A.dtype in (torch.float16, torch.bfloat16):
            if A.dtype == torch.bfloat16:
                B = _as_bfloat16_scalar(B)
            return floor_div_lowp_tensor_scalar_func(A, B, out0=A)
        return floor_div_func_corrected_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


# Large-tensor codegen variants (shrinking DMA pipeline / unroll config):
# the default-config kernels are launch-optimal for small tensors and are
# kept for numel < REMAINDER_CFG_THRESHOLD, where the config_ variants were
# measured to be up to ~3x slower (launch-floor cells).
REMAINDER_CFG_THRESHOLD = 1 << 20


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def rem_tt_cfg(x, y):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def rem_ts_cfg(x, y):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def rem_st_cfg(x, y):
    return _remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if max(A.numel(), B.numel()) >= REMAINDER_CFG_THRESHOLD:
            return rem_tt_cfg(A, B)
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        if A.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_ts_cfg(A, B)
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_st_cfg(A, B)
        return rem_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_KUNLUNXIN REMAINDER_")
    if isinstance(B, torch.Tensor):
        if max(A.numel(), B.numel()) >= REMAINDER_CFG_THRESHOLD:
            return rem_tt_cfg(A, B, out0=A)
        return rem_tt(A, B, out0=A)
    else:
        if A.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_ts_cfg(A, B, out0=A)
        return rem_ts(A, B, out0=A)
