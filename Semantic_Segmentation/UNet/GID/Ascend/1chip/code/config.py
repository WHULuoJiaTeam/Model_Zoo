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

from easydict import EasyDict as ed

config = ed({
    "device_target":"Ascend",      #GPU \ CPU \ Ascend
    "device_id":0,  #显卡ID
    "dataset_path": "/cache/data",  #数据存放位置
    "save_checkpoint_path": "/cache/checkpoint",  #保存的参数存放位置
})
