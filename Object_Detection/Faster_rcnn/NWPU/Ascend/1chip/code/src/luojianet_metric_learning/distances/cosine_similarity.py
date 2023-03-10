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

# from .dot_product_similarity import DotProductSimilarity
from ..distances.dot_product_similarity import DotProductSimilarity


class CosineSimilarity(DotProductSimilarity):
    # checked forward(), compute_mat() and pairwise_distance()
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings


if __name__ == "__main__":
    import numpy as np
    import luojianet_ms
    import torch
    import luojianet_ms.ops as P

    import luojianet_ms.context as context
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

    embedding_1 = np.array([[1.0, 2, 3], [4, 5, 6]])
    a = luojianet_ms.Tensor(embedding_1, dtype=luojianet_ms.float32)
    embedding_2 = np.array([[1.0, 1, 3], [4, 5, 6]])
    b = luojianet_ms.Tensor(embedding_2, dtype=luojianet_ms.float32)

    comput_cos_dis = CosineSimilarity()

    # s = torch.sum(torch.from_numpy(embedding_1) * torch.from_numpy(embedding_2), dim=1)
    # s = torch.max(torch.from_numpy(embedding_1))
    # s = torch.matmul(torch.from_numpy(embedding_1), torch.from_numpy(embedding_2).t())
    # s = torch.mean(torch.from_numpy(embedding_1))
    # print(s)

    # r = np.sum(a.asnumpy()*b.asnumpy(),  axis=1)
    # r = luojianet_ms.Tensor.from_numpy(r)
    # r = max(a.asnumpy())
    # r = a[r]
    # r = P.matmul(a, b.T)
    # r = P.ReduceSum(keep_dims=True)(a * b)
    # r = luojianet_ms.ops.ReduceMean()(a)
    # print(r)

    output = comput_cos_dis(a, b)
    print(output)



