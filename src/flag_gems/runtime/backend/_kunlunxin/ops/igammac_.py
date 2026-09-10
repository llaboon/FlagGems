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

logger = logging.getLogger(__name__)

# tests/test_igammac_.py asserts "GEMS SPECIAL_GAMMAINCC" on the generic op's
# logger name "flag_gems.ops.special_gammaincc", so the override must emit the
# exact same record instead of the _kunlunxin module logger.
_special_gammaincc_logger = logging.getLogger("flag_gems.ops.special_gammaincc")

_SUPPORTED_DTYPES = (torch.float32, torch.float64)

_SERIES_ITERS = 50
_CF_ITERS = 50
_BLOCK_SIZE = 256


@triton.jit
def _lgamma_impl(z):
    # Pure-Triton Lanczos log-gamma (g=7, n=8) with reflection for z < 0.5.
    # The XPU libdevice `lgamma` is an "Unsupported" stub that fails to link,
    # so log-gamma is computed here instead of via tl_extra_shim.lgamma.
    pi = 3.141592653589793
    half_log_2pi = 0.9189385332046727
    g = 7.0

    reflect = z < 0.5
    zr = tl.where(reflect, 1.0 - z, z)
    zz = zr - 1.0

    x = 0.99999999999980993
    x += 676.5203681218851 / (zz + 1.0)
    x += -1259.1392167224028 / (zz + 2.0)
    x += 771.32342877765313 / (zz + 3.0)
    x += -176.61502916214059 / (zz + 4.0)
    x += 12.507343278686905 / (zz + 5.0)
    x += -0.13857109526572012 / (zz + 6.0)
    x += 9.9843695780195716e-6 / (zz + 7.0)
    x += 1.5056327351493116e-7 / (zz + 8.0)

    t = zz + g + 0.5
    result = half_log_2pi + (zz + 0.5) * tl.log(t) - t + tl.log(x)

    reflected = tl.log(pi) - tl.log(tl.sin(pi * z)) - result
    return tl.where(reflect, reflected, result)


@triton.jit
def _gammaincc_q(a_f, x_f, SERIES_ITERS: tl.constexpr, CF_ITERS: tl.constexpr):
    # Q(a, x) = Gamma(a, x) / Gamma(a) (regularized upper incomplete gamma).
    log_gamma_a = _lgamma_impl(a_f)
    log_x_term = a_f * tl.log(x_f) - x_f - log_gamma_a

    # Power series for the lower regularized gamma P(a, x), then Q = 1 - P.
    # Converges for x < a + 1.
    term = 1.0 / a_f
    series_sum = term
    for i in range(1, SERIES_ITERS):
        term = term * x_f / (a_f + tl.cast(i, tl.float32))
        series_sum = series_sum + term
    q_series = 1.0 - tl.exp(log_x_term) * series_sum

    # Lentz's continued fraction for Q(a, x) directly. Converges for x >= a + 1.
    tiny = 1e-30
    b0 = x_f + 1.0 - a_f
    f_val = b0
    c_val = b0
    d_val = tl.zeros_like(x_f)
    for i in range(1, CF_ITERS):
        i_f = tl.cast(i, tl.float32)
        an = i_f * (a_f - i_f)
        bn = x_f + 2.0 * i_f + 1.0 - a_f

        d_val = bn + an * d_val
        d_val = tl.where(tl.abs(d_val) < tiny, tiny, d_val)
        c_val = bn + an / c_val
        c_val = tl.where(tl.abs(c_val) < tiny, tiny, c_val)

        d_val = 1.0 / d_val
        delta = c_val * d_val
        f_val = f_val * delta
    q_cf = tl.exp(log_x_term - tl.log(f_val))
    q_cf = tl.where(q_cf > 1.0, 1.0, tl.where(q_cf < 0.0, 0.0, q_cf))

    return tl.where(x_f < a_f + 1.0, q_series, q_cf)


@triton.jit
def gammaincc_kernel(
    a_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SERIES_ITERS: tl.constexpr,
    CF_ITERS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    q = _gammaincc_q(a, x, SERIES_ITERS, CF_ITERS)

    # Out-of-domain (a <= 0 or x < 0) -> NaN; Q(a, 0) = 1 for a > 0.
    q = tl.where((a <= 0.0) | (x < 0.0), float("nan"), q)
    q = tl.where((x == 0.0) & (a > 0.0), 1.0, q)

    tl.store(out_ptr + offsets, q, mask=mask)


def _launch(out, a, x):
    n = out.numel()
    if n == 0:
        return
    a_c = a.contiguous()
    x_c = x.contiguous()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        gammaincc_kernel[grid](
            a_c,
            x_c,
            out,
            n,
            BLOCK_SIZE=_BLOCK_SIZE,
            SERIES_ITERS=_SERIES_ITERS,
            CF_ITERS=_CF_ITERS,
        )


def igammac_(A, B):
    logger.debug("GEMS_KUNLUNXIN IGAMMAC_")
    if A.dtype not in _SUPPORTED_DTYPES or B.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"igammac_ Triton kernel supports dtypes {_SUPPORTED_DTYPES}, "
            f"but got A.dtype={A.dtype}, B.dtype={B.dtype}"
        )

    b = B if B.shape == A.shape else B.expand(A.shape)
    if A.numel() == 0:
        return A

    # Compute into a fresh float32 buffer, then write back in-place so that
    # non-contiguous / float64 inputs are handled uniformly.
    out = torch.empty(A.shape, dtype=torch.float32, device=A.device)
    _launch(out, A, b)
    A.copy_(out)
    return A


def special_gammaincc(self, other):
    _special_gammaincc_logger.debug("GEMS SPECIAL_GAMMAINCC")
    if self.dtype not in _SUPPORTED_DTYPES or other.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"special_gammaincc Triton kernel supports dtypes {_SUPPORTED_DTYPES}, "
            f"but got self.dtype={self.dtype}, other.dtype={other.dtype}"
        )

    a, x = torch.broadcast_tensors(self, other)
    out = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    _launch(out, a, x)

    result_dtype = torch.promote_types(self.dtype, other.dtype)
    if result_dtype != torch.float32:
        out = out.to(result_dtype)
    return out
