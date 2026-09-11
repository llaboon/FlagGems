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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Same config as the sibling exp.py (cosh is an exp-family op): vec-closed +
# kunlunAutoGrid + unroll8 (isCloseVectorization=True is the log/exp-family
# recipe; atanh/arcsinh use False but both are verified-correct on the full
# test matrix, and True is the safer choice for the exp family).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def cosh_func(x):
    x = x.to(tl.float32)
    return 0.5 * (tl.exp(x) + tl.exp(-x))


# cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def cosh_out(A, out):
    logger.debug("GEMS_KUNLUNXIN COSH_OUT")
    return cosh_func(A, out0=out)
