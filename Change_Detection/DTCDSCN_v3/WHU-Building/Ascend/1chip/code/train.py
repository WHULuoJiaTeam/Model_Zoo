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
import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.train import Model
from luojianet_ms.common import set_seed
from luojianet_ms.dataset import config
from luojianet_ms.train.callback import TimeMonitor, LossMonitor
from luojianet_ms import load_checkpoint, load_param_into_net
from luojianet_ms.train.callback import CheckpointConfig, ModelCheckpoint
from dataset import create_Dataset
from module import CDNet34
from config import config 
from cdloss import cdloss
import moxing as mox

import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    # obs download
    mox.file.copy_parallel(config.obs_dataset_path,
                           config.dataset_path)

    '''trian'''
    context.set_context(mode=context.PYNATIVE_MODE,device_target=config.device_target,device_id=config.device_id)

    train_dataset, config.steps_per_epoch = create_Dataset(config.dataset_path, config.aug, config.batch_size, shuffle=True)
    net = CDNet34(3)
    
    if config.resume:
        ckpt = load_checkpoint('**.ckpt')
        load_param_into_net(net, ckpt)

    lr = nn.cosine_decay_lr(min_lr=config.min_lr,max_lr=config.max_lr,total_step=config.epoch_size,step_per_epoch=config.steps_per_epoch,decay_epoch=config.decay_epochs)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    time_cb = TimeMonitor(data_size=config.steps_per_epoch)
    loss_cb = LossMonitor(168)
    loss = cdloss()
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'})

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model' + '/')
    ckpoint_cb = ModelCheckpoint(prefix="DTCDSCN", directory=save_ckpt_path, config=config_ck)
    callbacks = [time_cb, loss_cb, ckpoint_cb]
    
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks)
    print("============== Training Finished ==============")

    # obs uploads
    mox.file.copy_parallel(config.save_checkpoint_path,
                           'obs://luojianet-benchmark/Change_Detection/DTCDSCN_v3/WHU-BCD/1chip/ckpt/')