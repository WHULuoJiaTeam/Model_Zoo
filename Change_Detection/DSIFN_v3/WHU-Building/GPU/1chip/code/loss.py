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
import luojianet_ms.nn as nn
import luojianet_ms as ms
from luojianet_ms import ops

def cd_loss(inputt,target):

    # print(inputt.shape)
    # print(target.shape)

    #判断pred和target的维度，若不一致，则增维
    ndim_pred = len(inputt.shape)
    ndim_target = len(target.shape)
    if ndim_target>ndim_pred:
        target = target.reshape(inputt.shape)
        
    bce_loss = nn.BCELoss()
    sigmoid = ops.Sigmoid()
    bce_loss = bce_loss(inputt,target)

    smooth = 1.
    iflat = inputt.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dic_loss = 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))
    return  (dic_loss + bce_loss).mean()


class loss_fusion(LossBase):
    def __init__(self):
        super(loss_fusion,self).__init__()
        self.rb = nn.ResizeBilinear()

    def forward(self, logit, label):
        batch_loss1 = cd_loss(logit[0], label) #labe 512,512
        label_rz_branch_out2 = self.rb(label,size=(256,256), align_corners=True )
        # logit bs,1,512,512
        batch_loss2 = cd_loss(logit[1], label_rz_branch_out2) # label input -> 256, 256
        label_rz_branch_out3 = self.rb(label,size=(128,128), align_corners=True )
        batch_loss3 = cd_loss(logit[2], label_rz_branch_out3)
        label_rz_branch_out4 = self.rb(label,size=(64,64), align_corners=True )
        batch_loss4 = cd_loss(logit[3], label_rz_branch_out4)
        label_rz_branch_out5 = self.rb(label,size=(32,32), align_corners=True )
        batch_loss5 = cd_loss(logit[4], label_rz_branch_out5)
        Loss = (batch_loss1 + batch_loss2 +batch_loss3 +batch_loss4 +batch_loss5).mean()
        return Loss


