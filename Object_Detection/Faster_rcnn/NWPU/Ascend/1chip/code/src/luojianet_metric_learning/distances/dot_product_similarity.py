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

import luojianet_ms.ops as P
from luojianet_ms import Tensor

# from .base_distance import BaseDistance
from ..distances.base_distance import BaseDistance
import numpy as np


class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):  # function checked by xwj
        # checked by xwj
        # return torch.matmul(query_emb, ref_emb.t())
        return P.matmul(query_emb, ref_emb.T)

    def pairwise_distance(self, query_emb, ref_emb):  # function checked by xwj
        # checked by xwj
        r = np.sum(query_emb.asnumpy() * ref_emb.asnumpy(), axis=1)
        return Tensor.from_numpy(r)
        # return torch.sum(query_emb * ref_emb, dim=1)



