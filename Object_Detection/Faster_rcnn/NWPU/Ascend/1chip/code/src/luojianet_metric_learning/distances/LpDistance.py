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
import luojianet_ms
# from torch import cdist, from_numpy
from scipy.spatial import distance_matrix

# from ..utils import loss_and_miner_utils as lmu
from src.luojianet_metric_learning.utils import loss_and_miner_utils as lmu
# from .base_distance import BaseDistance
from src.luojianet_metric_learning.distances.base_distance import BaseDistance


class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):

        dtype = query_emb.dtype
        if ref_emb is None:
            ref_emb = query_emb

        rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)

        output = luojianet_ms.ops.Zeros()(rows.shape, dtype)
        rows, cols = rows.flatten(), cols.flatten()
        distances = self.pairwise_distance(query_emb[rows], ref_emb[cols]).astype(np.float32)
        output[rows, cols] = luojianet_ms.Tensor(distances, dtype=luojianet_ms.float32)
        return output

    def pairwise_distance(self, query_emb, ref_emb):

        return distance_matrix(query_emb.asnumpy(), ref_emb.asnumpy(), p=self.p).diagonal()


if __name__ == "__main__":
    import numpy as np
    import luojianet_ms
    import luojianet_ms.nn as nn

    import luojianet_ms.context as context
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    embedding_1 = np.array([[1.0, 2, 3], [4, 5, 6], [4, 5, 6]])
    a = luojianet_ms.Tensor(embedding_1, dtype=luojianet_ms.float64)
    embedding_2 = np.array([[1.0, 1, 3], [4, 5, 6], [4, 5, 6]])
    b = luojianet_ms.Tensor(embedding_2, dtype=luojianet_ms.float64)

    comput_cos_dis = LpDistance(p=2)

    output = comput_cos_dis.compute_mat(a, b)
    print(output)

