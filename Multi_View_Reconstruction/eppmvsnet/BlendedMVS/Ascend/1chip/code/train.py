# Copyright 2021 Huawei Technologies Co., Ltd
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
"""EPP-MVSNet's validation process on BlendedMVS dataset"""

import os
import time
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm

import luojianet_ms.dataset as ds
from luojianet_ms import context
from luojianet_ms.ops import operations as P
from luojianet_ms import nn
from luojianet_ms import dtype as mstype
from luojianet_ms import Model
from luojianet_ms.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.eppmvsnet import EPPMVSNet
from src.blendedmvs import BlendedMVSDataset
from src.loss import EPPMVSNetWithLoss

# from luojianet_ms.communication.management import init,get_rank,get_group_size

class L1Loss(nn.LossBase):
    def __init__(self, mask_thre=0):
        super(L1Loss, self).__init__()
        self.abs = P.Abs()
        self.mask_thre = mask_thre

    def forward(self, predict, label):
        mask = (label >= self.mask_thre).astype(mstype.float32)
        num = mask.shape[0] * mask.shape[1] * mask.shape[2]
        x = self.abs(predict * mask - label * mask)

        return self.get_loss(x) / P.ReduceSum()(mask) * num

def get_opts():
    """set options"""
    parser = ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='which gpu used to inference')
    ## data
    parser.add_argument('--root_dir', type=str,
                        default='D:\\eppmvsnet\\BlendedMVS',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='blendedmvs',
                        choices=['blendedmvs'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default='train',
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default=None, nargs='+',
                        help='specify scan to evaluate (must be in the split)')
    # for depth prediction
    parser.add_argument('--n_views', type=int, default=5,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--depth_interval', type=float, default=128,
                        help='depth interval unit in mm')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[32, 16, 8],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[4.0, 2.0, 1.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1152, 864],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/exp2/_ckpt_epoch_10.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_visual', default=False, action='store_true',
                        help='save depth and proba visualization or not')
    parser.add_argument('--entropy_range', action='store_true', default=False,
                        help='whether to use entropy range method')
    parser.add_argument('--conf', type=float, default=0.9,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--levels', type=int, default=3, choices=[3, 4, 5],
                        help='number of FPN levels (fixed to be 3!)')
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--amp_level", type=str, default='O0', help="amp level")
    parser.add_argument("--epochs", type=int, default=10, help="the number of epoch")
    parser.add_argument('--ckpt_dir', type=str,
                        default='D:\\eppmvsnet\\BlendedMVS',
                        help='root directory of ckpt')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args.gpu_id, save_graphs=False,
                        enable_graph_kernel=False)
    # init()
    dataset = BlendedMVSDataset(args.root_dir, args.split, n_views=args.n_views, depth_interval=args.depth_interval,
                                img_wh=tuple(args.img_wh), levels=args.levels, scan=args.scan)

    print(args.n_depths)
    print(args.interval_ratios)
    print(args)
    # Step 1. Create model
    EPPMVSNet_train = EPPMVSNet(n_depths=args.n_depths, interval_ratios=args.interval_ratios,
                               entropy_range=args.entropy_range, height=args.img_wh[1], width=args.img_wh[0])
    EPPMVSNet_train.set_train(True)
    NetWithLoss = EPPMVSNetWithLoss(EPPMVSNet_train)

    # Step 2. Create loss
    loss_func = L1Loss()

    # Step 3. Choose optimizer
    net_opt = nn.RMSProp(EPPMVSNet_train.trainable_params(), learning_rate=args.lr)

    # Step 4. Creat model
    model = Model(EPPMVSNet_train, loss_fn=None, optimizer=net_opt, amp_level=args.amp_level)


    train_loader = ds.GeneratorDataset(dataset, column_names=["imgs", "proj_mats", "init_depth_min", "depth_interval",
                                                             "scan", "vid", "depth_0", "mask_0", "fix_depth_interval"],
                                      num_parallel_workers=1, shuffle=False)
    train_loader = train_loader.batch(batch_size=1)
    train_data_size = train_loader.get_dataset_size()
    print("train dataset length is:", train_data_size)

    # Step 6. save checkpoint of the model
    config_ck = CheckpointConfig(save_checkpoint_steps=train_loader.get_dataset_size(), keep_checkpoint_max=args.epochs)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_eppmvsnet_whu", directory=args.ckpt_dir, config=config_ck)
    time_cb = TimeMonitor()

    output = model.train(args.epochs, train_loader, callbacks=[ckpoint_cb, LossMonitor(1), time_cb],
                         dataset_sink_mode=False)

