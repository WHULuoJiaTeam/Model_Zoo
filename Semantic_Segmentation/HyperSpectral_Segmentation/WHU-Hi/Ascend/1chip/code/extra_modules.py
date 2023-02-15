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

import luojianet_ms.nn as nn
import luojianet_ms as ms
import luojianet_ms.dataset as ds


class LossCell(nn.Module):
    def __init__(self, backbone, loss_fn):
        super(LossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn
        
    def forward(self, data, label):
        # out = self.backbone(data, label)
        out = self.backbone(data)
        return self.loss_fn(out, label)
        
    def backbone_network(self,):
        return self.backbone

        
class TrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(TrainStep, self).__init__(network, optimizer)
        self.grad = ms.ops.GradOperation(get_by_list=True)
        
        
    def forward(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)
