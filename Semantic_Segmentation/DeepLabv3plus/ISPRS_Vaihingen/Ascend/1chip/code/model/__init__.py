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
from .deeplabv3p import DeepLabV3Plus

def get_deeplabv3p(in_channels, n_class):
    return DeepLabV3Plus(num_classes=n_class, output_stride=16,aspp_atrous_rates=[1, 12, 24, 36])

class SegModel(nn.Module):
    def __init__(self,model_network:str,in_channels: int = 3, n_class: int = 6):
        super(SegModel, self).__init__()
        self.in_channels=in_channels
        self.n_class=n_class
        self.model_network=model_network
        self.model=get_deeplabv3p(in_channels, n_class)
    def forward(self,x):
        x =self.model(x)
        return x
