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

from luojianet_ms.nn.loss.loss import LossBase

import luojianet_ms as ms
from luojianet_ms import ops, nn

class cdloss(LossBase):
    def __init__(self, gamma = 1.5, size_average=True):
        super(cdloss,self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, label):
        Loss = 0
        for i in range(3):
            target = label[:,i,:,:,:]
            prob = logit[i]
            target = target.view(-1)
            prob = prob.view(-1)
            prob_p = ops.clip_by_value(prob, 1e-8, 1 - 1e-8)
            prob_n = ops.clip_by_value(1.0 - prob, 1e-8, 1 - 1e-8)
            batch_loss= - ops.Pow()((2 - prob_p),self.gamma) * ops.log(prob_p)* target \
                        - ops.log(prob_n) * (1 - target) *(2 - prob_n)

            Loss += batch_loss
        if self.size_average:
            loss = Loss.mean()
        else:
            loss = Loss
        return loss