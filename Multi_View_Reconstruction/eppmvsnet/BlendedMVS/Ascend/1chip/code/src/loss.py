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

from luojianet_ms import nn
from luojianet_ms import dtype as mstype
from luojianet_ms.ops import operations as P
import numpy as np

class L1Loss(nn.LossBase):
    def __init__(self, mask_thre=0):
        super(L1Loss, self).__init__()
        self.abs = P.Abs()
        self.mask_thre = mask_thre

    def forward(self, predict, label):
        mask = (label >= self.mask_thre).astype(mstype.float32)
        num = mask.shape[0] * mask.shape[1] * mask.shape[2]
        x = self.abs(predict * mask - label * mask)

        return self.get_loss(x) / P.ReduceSum()(mask) * num

class EPPMVSNetWithLoss(nn.Module):
    def __init__(self, network, loss=L1Loss()):
        super(EPPMVSNetWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss

    def forward(self,imgs, proj_mats, init_depth_min, depth_interval, scan, vid, depth_0, mask_0, fix_depth_interval):
        gt = P.Squeeze()(depth_0).asnumpy()
        mask = P.Squeeze()(mask_0).asnumpy()
        depth, _ = self.network(imgs, proj_mats, init_depth_min, depth_interval)
        depth = P.Squeeze()(depth).asnumpy()
        depth = np.nan_to_num(depth)  # change nan to 0

        abs_err = np.abs(depth - gt)
        abs_err_scaled = abs_err / fix_depth_interval.asnumpy()

        loss = abs_err_scaled[mask].mean()

        return loss

    def backbone_network(self):
        return self.network


