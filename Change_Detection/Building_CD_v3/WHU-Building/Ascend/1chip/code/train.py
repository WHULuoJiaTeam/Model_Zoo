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

import os
import numpy as np
import random
from PIL import Image
from mainnet import *
import luojianet_ms
from luojianet_ms.dataset.vision import Inter
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision
import luojianet_ms.dataset as ds
import cv2
from luojianet_ms import context,nn
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from luojianet_ms import load_checkpoint, load_param_into_net
from dataset import create_Dataset
from config import config 

import moxing as mox

class cross_entropy(nn.Module):
    def __init__(self):
        super(cross_entropy, self).__init__()
        self.meanloss = luojianet_ms.ops.ReduceMean()

    def forward(self, prediction, label):
        return -self.meanloss(label * luojianet_ms.ops.log(prediction) + (1 - label) * luojianet_ms.ops.log(1 - prediction))

if __name__ == '__main__':
    # obs download
    mox.file.copy_parallel(config.obs_dataset_path,
                           config.dataset_path)

    context.set_context(mode=context.GRAPH_MODE,device_target=config.device_target,device_id=config.device_id)
    Datasets, config.steps_per_epoch = create_Dataset(config.dataset_path, config.aug, config.batch_size, shuffle=True)

    net = two_net()

    if config.resume:
        ckpt = load_checkpoint('**.ckpt')
        load_param_into_net(net, ckpt)

    optimizer = nn.Adam(net.trainable_params(), learning_rate=config.LR)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs, keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix='CD', directory=config.save_checkpoint_path, config=config_ck)
    model = luojianet_ms.Model(net, loss_fn=cross_entropy(), optimizer=optimizer)
    print("============== Starting Training ==============")
    model.train(config.epoch_size, Datasets, callbacks=[ckpt_cb, LossMonitor(200)],dataset_sink_mode=True)
    print("============== Training Finished ==============")

    # obs uploads
    mox.file.copy_parallel(config.save_checkpoint_path,
                           'obs://luojianet-benchmark/Change_Detection/Building_CD_v3/WHU-BCD/1chip/ckpt/')