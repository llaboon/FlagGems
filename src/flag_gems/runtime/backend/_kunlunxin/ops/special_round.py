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

# torch.special.round(input, *, decimals) is an alias of torch.round
# (round-half-to-even, identical semantics). On XPU the generic
# flag_gems/ops/special_round.py kernel is dominated by a per-element fp64
# division emulation (fp64_enabled=False on this backend) plus libdevice floor
# extern calls, e.g. ~11s on 16M-element fp16 vs ~40us for torch. The vendor
# _kunlunxin.ops.round fast paths are proven bit-exact vs CPU reference on the
# same round-half-to-even matrix, so special_round delegates to them:
#   - decimals == 0: magic-number RN  (x + 1.5*2^23) - 1.5*2^23, pure fp32
#     add/sub, no extern call and no division;
#   - decimals != 0: libdevice rint(x * 10^d) / 10^d in fp32.

import logging

from .round import round as _round
from .round import round_out as _round_out

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def special_round(input, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ROUND")
    return _round(input, decimals=decimals)


def special_round_out(input, out, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ROUND_OUT")
    return _round_out(input, decimals=decimals, out=out)