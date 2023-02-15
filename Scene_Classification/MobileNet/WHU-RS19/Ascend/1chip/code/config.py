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
You can forward the following directory structure from your dataset files and read by LuoJiaNet's API.

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── class1
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class2
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class3
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── classN
             ├── ...
'''
config = ed({
    "device_target":"Ascend",      #GPU或CPU
    "dataset_path": "obs://sktest/WHU-RS19-train/",  #数据存放位置
    "dataset_eval_path": "obs://sktest/WHU-RS19-test/",  #验证数据存放位置
    "save_checkpoint_path": "/cache/training/",  #保存的参数存放位置
    "obs_checkpoint_path": "obs://luojianet-benchmark/Scene_Classification/MobileNet/WHURS-19/1chip/ckpt/",
    "resume":False,   #是否载入模型训练
    "class_num": 19,  #数据集中包含的种类
    "batch_size": 64,
    "loss_scale": 1024,
    "momentum": 0.9,
    "alpha": 1, # mobilenet的alpha参数
    "weight_decay": 1e-5,
    "epoch_size": 200, #训练次数
    "save_checkpoint": True, #是否保存模型
    "save_checkpoint_epochs": 1, #多少次迭代保存一次模型
    "keep_checkpoint_max": 100, #文件内保存模型的最大个数，超过则删掉最开始的
    "opt": 'rmsprop', #优化器：rmsprop或sgd
    "opt_eps": 0.001,
    "warmup_epochs": 50, #warmup训练策略
    "lr_decay_mode": "warmup", #学习率衰减方式：steps、poly、cosine以及warmup
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.0001, #初始学习率
    "lr_max": 0.001, #最大学习率
    "lr_end": 0.00001 #最小学习率
})
