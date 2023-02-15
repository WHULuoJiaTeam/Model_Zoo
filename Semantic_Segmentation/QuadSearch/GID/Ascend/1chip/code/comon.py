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
from luojianet_ms import ops
class MyWithLossCell(nn.Module):
    """定义损失网络"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def call(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label), out

    def backbone_network(self):
        return self.backbone


class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def call(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss, out = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads), out

class MyWithEvalCell(nn.Module):
    """定义验证流程"""
    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def call(self, data, label):
        outputs = self.network(data)
        return outputs, label