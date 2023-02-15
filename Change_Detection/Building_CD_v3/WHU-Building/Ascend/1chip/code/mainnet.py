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

from vgg import *
from fpn import *
from finalnet import *

#import matplotlib.pyplot as plt
#import seaborn as sns

class two_net(nn.Module):
    def __init__(self):
        super(two_net, self).__init__()
        self.vgg = FCNnet()
        self.fpn1 = fpn()
        self.finalnet1 = finalnet()
        #self.conv = nn.Conv2d(64,1,kernel_size=3,stride=1, padding=1)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.cat = luojianet_ms.ops.Concat(axis=1)


    def forward(self, data):
        x1 = data[:, 0:3, :, :]
        x2 = data[:, 3:6, :, :]
        p2_1, p3_1, p4_1, p5_1, pa = self.vgg(x1)
        p2_2, p3_2, p4_2, p5_2, pb = self.vgg(x2)


        pd_2, pd_3, pd_4, pd_5 = p2_1 - p2_2, p3_1 - p3_2, p4_1 - p4_2, p5_1 - p5_2
        pd_2, pd_3, pd_4, pd_5 = self.fpn1(pd_2, pd_3, pd_4, pd_5)
        
        pd = self.finalnet1(pd_2, pd_3, pd_4, pd_5)

        return self.cat((pa, pb, pd))



