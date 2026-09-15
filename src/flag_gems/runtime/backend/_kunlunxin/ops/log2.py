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

# Kunlunxin (XPU) override of aten.log2 (functional).
#
# Two independent defects fixed here relative to the generic flag_gems/ops/log2.py:
#
# 1) CORRECTNESS — tl.log2 is base-e on this XPU (returns ln(x), NOT log2(x)).
#    This was isolated in the logaddexp2 fix (see _kunlunxin/ops/logaddexp2.py):
#    on this backend `tl.log2(z)` returns ln(z). The generic op computes
#    tl.log2(x.to(f32)), which therefore returns ln(x), off by an ln(2) factor
#    (~99.8% mismatch, max rel ~0.31 vs an fp64 CPU reference). Rebuild the
#    base-2 result from the natural-base primitive:
#        log2(x) = ln(x) / ln(2) = ln(x) * 1.4426950408889634.
#
# 2) PERFORMANCE — the generic op decorates the kernel with the bare
#    pointwise_dynamic (no CodeGenConfig), so on XPU it is specialized per shape
#    -> per-shape recompile / IR explosion and discrete access. Baseline:
#    fp16 [4096,4096] ~42ms vs ~0.18ms torch (speedup ~0.004). Fix = the standard
#    memory-bound unary CodeGenConfig (kunlunAutoGrid=True, prefer_1d_tile,
#    buffer_size_limit=4096, unroll_num=8) so the kernel is shape-independent,
#    compiled once, and does contiguous block DMA. Mirrors acosh / log10_ / exp2.
#
# isCloseVectorization=False (vectorization OPEN) matching acosh/log10_: log2 =
# log(x)*inv_ln2 is a transcendental kernel; the bf16 vectorized tl.log
# miscompile seen with log1p's `1.0 + x` addend is absent here.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# 1 / ln(2) = 1.4426950408889634 (fp32 rel. error ~1.2e-7, far below the test
# rtol of 1.3e-6 for fp32 / 1e-3 for fp16/bf16). Inlined literal: triton jit
# only allows constexpr globals.

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


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=config_)
@triton.jit
def log2_func(x):
    # log2(x) = ln(x) / ln(2); tl.log2 is base-e on this backend, so use tl.log.
    return tl.log(x.to(tl.float32)) * 1.4426950408889634


def log2(A):
    logger.debug("GEMS_KUNLUNXIN LOG2")
    return log2_func(A)


def log2_(A):
    logger.debug("GEMS_KUNLUNXIN LOG2 INPLACE")
    log2_func(A, out0=A)
    return A