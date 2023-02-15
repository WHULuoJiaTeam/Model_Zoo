# Copyright 2021, 2022, 2023 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022, 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# import torch
import luojianet_ms.ops as P

from .base_reducer import BaseReducer


class MeanReducer(BaseReducer):
    def element_reduction(self, losses, *_):
        # check by xwj
        # return torch.mean(losses)
        return P.reduce_mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)
