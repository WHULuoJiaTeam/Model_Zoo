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

'''
You can call the following directory structure from your dataset files and read by LuoJiaNet's API.

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── A
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── B
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── label
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
'''
config = ed({
    "device_target": "GPU",  # GPU \ CPU \ Ascend
    "device_id": 0,  # 显卡ID
    "dataset_path": "/mnt/SourceCodes/ZhouHuanCode/DATASET/WHU-BCD",  # 数据存放位置
    "save_checkpoint_path": "./checkpoint/",  # 保存的参数存放位置
    "resume": False,  # 是否载入模型训练
    "batch_size": 4,
    "aug": True,
    "steps_per_epoch": 200,
    "epoch_size": 200,  # 训练次数
    "save_checkpoint": True,  # 是否保存模型
    "save_checkpoint_epochs": 200,  # 多少次迭代保存一次模型
    "keep_checkpoint_max": 10,  # 保存模型的最大个数
    "decay_epochs": 20,  # 学习率衰减的epoch数
    "max_lr": 0.001,  # 最大学习率
    "min_lr": 0.00001  # 最小学习率
})
