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

import luojianet_ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
import luojianet_ms.numpy as mnp
from luojianet_ms import Tensor


class MVSLoss(nn.LossBase):
    def __init(self, reduction='mean'):
        super(MVSLoss, self).__init__(reduction)
        self.smooth_l1 = ops.SmoothL1Loss()

    def forward(self, logits, labels):
        x = self.smooth_l1(logits, labels)
        output = self.get_loss(x)

        return output


class MVSNetWithLoss(nn.Module):
    def __init__(self, network, loss=ops.SmoothL1Loss()):
        super(MVSNetWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.reduce_sum = ops.ReduceSum()
        self.upsample = nn.ResizeBilinear()

    def forward(self, img, camera, target, values, mask):

        output = self.network(img, camera, values)
        output = self.upsample(output, size=(target.shape[1], target.shape[2])).squeeze(1)

        output_with_mask = output * mask
        target_with_mask = target * mask
        loss = self.loss(output_with_mask, target_with_mask)

        return loss

    def backbone_network(self):
        return self.network
