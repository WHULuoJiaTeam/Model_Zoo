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

import argparse
from src.dataset import MVSDatasetGenerator
from src.mvsnet import MVSNet
from src.loss import MVSNetWithLoss
import luojianet_ms.dataset as ds
import os
import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt
from luojianet_ms import nn, Model
from luojianet_ms.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, LearningRateScheduler
from luojianet_ms import context
import luojianet_ms.dataset.vision as vision
import luojianet_ms as ms
from luojianet_ms.train.loss_scale_manager import FixedLossScaleManager
from luojianet_ms.communication.management import init,get_rank,get_group_size

from luojianet_ms.nn import learning_rate_schedule
from luojianet_ms import load_checkpoint, load_param_into_net
from src.benchmark_callback import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='A LuoJiaNET Implementation of MVSNet')
parser.add_argument('--dataset', default='whu', help='select dataset')

parser.add_argument('--data_root', default='D:\\BaiduNetdiskDownload\\WHU_MVS_dataset\\train', help='train datapath')
parser.add_argument('--logdir', default='checkpoints_mvsnet', help='the directory to save checkpoints/logs')
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--ndepths', type=int, default=200, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=768, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=384, help='Maximum image height')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=0.25, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--adaptive_scaling', type=bool, default=True, help='Let image size to fit the network, including scaling and cropping')

# network architecture
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--save_ckpt_epochs', type=int, default=5, help='number of epochs to save ckpt')
parser.add_argument('--keep_checkpoint_max', type=int, default=100, help='number of epochs to keep ckpt')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate for lr adjustment')
parser.add_argument('--decay_step', type=int, default=5000, help='decay step for lr adjustment')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()


def create_dataset(mode, args, rank_id, rank_size):

    ds.config.set_seed(args.seed)
    dataset_generator = MVSDatasetGenerator(args.data_root + 'train/', mode, args.view_num, args.normalize, args)

    input_data = ds.GeneratorDataset(dataset_generator,
                                     column_names=["image", "camera", "target", "values", "mask"],
                                     shuffle=False,
                                     num_shards=rank_size,
                                     shard_id=rank_id)
    input_data = input_data.batch(batch_size=args.batch_size)
    return input_data


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=int(os.environ["DEVICE_ID"]))
    init()
    #ms.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation", device_num=8, gradients_mean=True)
    rank_id = get_rank()
    rank_size = get_group_size()
    train_dataset = create_dataset("train", args, rank_id, rank_size)
    val_dataset = create_dataset("test", args, rank_id, rank_size)

    train_data_size = train_dataset.get_dataset_size()
    print(args)
    print(train_data_size)

    # create network
    net = MVSNet(args.max_h, args.max_w, args.batch_size, False)
    net_with_loss = MVSNetWithLoss(net)

    # learning rate and optimizer
    learning_rate = learning_rate_schedule.ExponentialDecayLR(args.lr, args.decay_rate, args.decay_step)
    net_opt = nn.RMSProp(net.trainable_params(), learning_rate=learning_rate)

    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)

    model = Model(net_with_loss, loss_fn=None, optimizer=net_opt, amp_level='O0',
                  loss_scale_manager=loss_scale, metrics={'0' : nn.Loss(), '1' : nn.Accuracy()})

    callbacks = [LossMonitor(per_print_times=10),
                 TimeMonitor(data_size=train_data_size),
                 BenchmarkTraining(model, val_dataset)]

    if rank_id == 0:
        time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_ckpt_epochs,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_mvsnet_whu", directory=args.logdir, config=config_ck)
        callbacks.append(ckpoint_cb)


    output = model.train(args.epochs, train_dataset, callbacks=callbacks,
                         dataset_sink_mode=False)

