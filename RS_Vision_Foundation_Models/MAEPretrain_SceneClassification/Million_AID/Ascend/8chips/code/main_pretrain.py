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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.common import set_seed
from luojianet_ms.communication.management import init, get_group_size, get_rank
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.ops import operations as P
from luojianet_ms import Tensor
from luojianet_ms import dtype as mstype
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import functional as F
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig
from luojianet_ms.train.model import Model
from luojianet_ms.train.callback import LossMonitor

import argparse
from pathlib import Path
import logging
import os
import sys
import datetime
from datetime import datetime
import time
import numpy as np
import math

from util.datasets import build_dataset
import models_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', default=False, action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # device context init parameters
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--mode', default='PYNATIVE_MODE', type=str, choices=['PYNATIVE_MODE', 'GRAPH_MODE'],
    #                     help='#0--Graph Mode; 1--Pynative Mode')
    parser.add_argument('--mode', default='GRAPH_MODE', type=str, choices=['PYNATIVE_MODE', 'GRAPH_MODE'],
                        help='#0--Graph Mode; 1--Pynative Mode')
    parser.add_argument("--device_target", default='Ascend', type=str, choices=['CPU', 'GPU', 'Ascend'])
    # parser.add_argument("--device_target", default='CPU', type=str, choices=['CPU', 'GPU', 'Ascend'])
    parser.add_argument('--max_call_depth', default=10000, type=int,
                        help='The depth of evoking function, if not satisfied, it will raise the error of core dumped.')
    parser.add_argument('--save_graphs', default=False, type=bool,
                        help='Whether to save computational graph. If it is Ture, should also add parameter of save_graphs_path.')
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--use_parallel', default=True, action='store_true')
    # parser.add_argument('--use_parallel', default=False, action='store_true')
    parser.add_argument("--parallel_mode", default='DATA_PARALLEL', type=str, choices=['DATA_PARALLEL', 'SEMI_AUTO_PARALLEL', 'AUTO_PARALLEL', 'HYBRID_PARALLEL'])
    parser.add_argument("--device_num", default=None, type=int,
                        help='number of devices')

    # dataset base parameters
    parser.add_argument('--dataset', default='millionaid', type=str, choices=['millionaid', 'imagenet'],
                        help='type of dataset')
    parser.add_argument("--tag", default=51, type=int,
                        help='different number of training samples')

    # train dataset parameters
    parser.add_argument('--data_path', default='D:/small_dataset_test/Million-AID', type=str,
                        help='dataset path')
    parser.add_argument('--num_workers', default=10, type=int)
    # parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # train parameters
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per device (effective batch size is batch_size * accum_iter * # devices')
    # parser.add_argument('--batch_size', default=2, type=int,
    #                     help='Batch size per device (effective batch size is batch_size * accum_iter * # devices')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--per_step_size', type=int, default=0,
                        help='clips values of multiple tensors by the ratio of the sum of their norms')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    # loss scale manager parameters
    parser.add_argument("--use_dynamic_loss_scale", default=True, type=bool,
                        help='Use in pretrain, whehter to use the strategy of dynamic loss scale')
    parser.add_argument("--loss_scale", default=1024, type=int,
                        help='use in finetune, the loss scale in FixedLossScaleUpdateCell of scale_manager')

    # with EMA parameters
    parser.add_argument("--use_ema", default=False, type=bool,
                        help='exponential Moving Average')
    parser.add_argument('--ema_decay', default=0.9999, type=float,
                        help='exponential Moving Average')

    # use_global_norm parameters
    parser.add_argument("--use_global_norm", default=False, type=bool,
                        help='the global norm of multiple tensors')
    parser.add_argument('--clip_gn_value', default=1.0, type=float,
                        help='clips values of multiple tensors by the ratio of the sum of their norms')

    # optimizer parameters
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).')
    parser.add_argument('--beta2', default=0.95, type=float,
                        help='The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).')

    # learning rate schedule parameters
    parser.add_argument('--lr', default=None, type=float, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', default=1e-3, type=float, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', default=0., type=float, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', default=40, type=int, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # ckpt init parameters
    parser.add_argument("--use_ckpt", default='', type=str,
                        help='pretrained models')
    parser.add_argument('--save_ckpt_epochs', default=1, type=int,
                        help='save checkpoint epochs')
    parser.add_argument('--keep_checkpoint_max', default=5, type=int,
                        help='number of saved checkpoint files')
    parser.add_argument("--prefix", default='MaePretrainViT-B-P16', type=str,
                        help='prefix of saved ckpt files')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save model, empty for no saving')

    return parser


PARALLEL_MODE = {'DATA_PARALLEL': context.ParallelMode.DATA_PARALLEL,
                 'SEMI_AUTO_PARALLEL': context.ParallelMode.SEMI_AUTO_PARALLEL,
                 'AUTO_PARALLEL': context.ParallelMode.AUTO_PARALLEL,
                 'HYBRID_PARALLEL': context.ParallelMode.HYBRID_PARALLEL}
MODE = {'PYNATIVE_MODE': context.PYNATIVE_MODE,
        'GRAPH_MODE': context.GRAPH_MODE}
def init_distributed_mode(args):
    """Init LuoJiaNET distributed configs and modes."""
    device_num = 1
    local_rank = 0
    if args.use_parallel:
        init()
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        local_rank = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        args.device_id = device_id
        context.set_context(mode=MODE[args.mode], device_target=args.device_target, device_id=args.device_id,
                            max_call_depth=args.max_call_depth, save_graphs=args.save_graphs)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=PARALLEL_MODE[args.parallel_mode], gradients_mean=True)
    else:
        print('Not using distributed mode')
        context.set_context(mode=MODE[args.mode], device_target=args.device_target, device_id=args.device_id,
                            max_call_depth=args.max_call_depth, save_graphs=args.save_graphs)

    os.environ['MOX_SILENT_MODE'] = '1'
    return local_rank, device_num


logger_name = 'luojianet_ms-benchmark'
class LOGGER(logging.Logger):
    """LOGGER"""
    def __init__(self, logger_name_local, rank=0):
        super().__init__(logger_name_local)
        self.log_fn = None
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', "%Y-%m-%d %H:%M:%S")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        """setup_logging_file"""
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(rank)
        log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.log_fn = log_fn

    def info(self, msg, *args, **kwargs):
        """info"""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        """save_args"""
        self.info('Args:')
        if isinstance(args, (list, tuple)):
            for value in args:
                message = '--> {}'.format(value)
                self.info(message)
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                message = '--> {}: {}'.format(key, args_dict[key])
                self.info(message)
        self.info('')


def get_logger(path, rank=0):
    """get_logger"""
    logger = LOGGER(logger_name, rank)
    logger.setup_logging_file(path, rank)
    return logger


def adjust_learning_rate(total_epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup.
    Reference:
        https://gitee.com/mindspore/models/blob/master/official/cv/DBNet/train.py
    """
    # for param in model.get_parameters():
    #     if not param.requires_grad:
    #         continue  # frozen weights
    #     if len(param.shape) == 1 or param.name.endswith('.bias') or param.name.endswith('.gamma') \
    #             or param.name.endswith('.beta') or param.name in skip_list:
    #         no_decay_params.append(param)
    #     else:
    #         decay_params.append(param)
    total_iters = total_epoch * args.per_step_size
    warmup_iters = args.warmup_epochs * args.per_step_size
    start_iters = args.start_epoch * args.per_step_size
    lr_each_iter = []
    for it in range(start_iters, total_iters):
        if it <= warmup_iters:
            lr_each_iter.append(args.lr * total_epoch / args.warmup_epochs)
        else:
            lr_each_iter.append(args.min_lr + (args.lr - args.min_lr) * 0.5 *
                       (1. + math.cos(math.pi * (total_epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))))
    return np.array(lr_each_iter, np.float32)


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    By using add_weight_decay(), nn.linear.bias, nn.LayerNorm.weight and nn.LayerNorm.bias will have weight_decay=0,
    and other parameters such as nn.Linear.weight will have weight_decay=args.weight_decay.
    Reference:
        https://github.com/rwightman/pytorch-image-models/discussions/1434
    """
    decay_params = []
    no_decay_params = []
    for param in model.get_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or param.name.endswith('.bias') or param.name.endswith('.gamma') \
                or param.name.endswith('.beta') or param.name in skip_list:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    group_params = [
        {'params': no_decay_params, 'weight_decay': 0.},
        {'params': decay_params, 'weight_decay': weight_decay}]
    return group_params


class EMACell(nn.Module):
    """EMACell Define"""
    def __init__(self, weights, ema_decay=0.9999):
        super(EMACell, self).__init__()
        self.ema_weights = weights.clone(prefix="_ema_weights")
        self.ema_decay = Tensor(ema_decay, mstype.float32)
        self.hyper_map = C.HyperMap()

    def forward(self, weights):
        success = self.hyper_map(F.partial(C.MultitypeFuncGraph("grad_ema_op"), self.ema_decay), self.ema_weights, weights)
        return success


class TrainOneStepWithClipGNAndEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer,
                 use_global_norm=False, clip_global_norm_value=1.0,
                 scale_sense=1.0, with_ema=False, ema_decay=0.9999):
        super(TrainOneStepWithClipGNAndEMA, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.with_ema = with_ema
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        if self.with_ema:
            self.ema_model = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(C.MultitypeFuncGraph("grad_scale"), scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            loss = F.depend(loss, self.optimizer(grads))
            if self.with_ema:
                self.ema_model(self.weights)
        else:
            self.print("=============Over Flow, skipping=============")
        return loss


def create_train_one_step(args, net_with_loss, optimizer, log):
    """
    get_train_one_step cell
    Reference:
        https://gitee.com/mindspore/models/blob/master/official/cv/MAE/src/trainer/trainer.py
    """
    if args.use_dynamic_loss_scale:
        log.info(f"=> Using DynamicLossScaleUpdateCell")
        scale_manager = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 24, scale_factor=2, scale_window=2000)
    else:
        log.info(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_manager = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)

    if args.use_ema and not args.use_global_norm:
        net_with_loss = TrainOneStepWithClipGNAndEMA(
            net_with_loss, optimizer, scale_sense=scale_manager,
            with_ema=args.use_ema, ema_decay=args.ema_decay)
    elif args.use_ema and args.use_global_norm:
        net_with_loss = TrainOneStepWithClipGNAndEMA(
            net_with_loss, optimizer, use_global_norm=args.use_global_norm,
            clip_global_norm_value=args.clip_gn_value, scale_sense=scale_manager,
            with_ema=args.use_ema, ema_decay=args.ema_decay)
    else:
        if args.device_target != 'CPU':
            net_with_loss = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scale_manager)
        else:
            print('luojianet_ms.nn.TrainOneStepWithLossScaleCell is not supported on CPU device.')
            net_with_loss = net_with_loss

    return net_with_loss


def main(args):
    """Training process."""
    # misc.init_distributed_mode(args)
    local_rank, device_num = init_distributed_mode(args)
    args.device_num = device_num
    args.local_rank = local_rank

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    seed = args.seed + local_rank
    set_seed(seed)

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None
    logger = get_logger(args.log_dir)
    logger.info("model config: {}".format(args))

    # train dataset
    if args.dataset == 'millionaid':
        # dataset_train = MillionAIDDataset(args.data_path, train=True, transform=transform_train, tag=args.tag)
        dataset_train = build_dataset(is_train=True, args=args)
    else:
        raise NotImplementedError

    data_size = dataset_train.get_dataset_size()
    new_epochs = args.epochs
    if args.per_step_size:
        new_epochs = int((data_size / args.per_step_size) * args.epoch)
    else:
        args.per_step_size = data_size
    logger.info("Will be Training epochs:{}, sink_size:{}".format(new_epochs, args.per_step_size))
    logger.info("Create training dataset finish, data size:{}".format(data_size))

    # output folder
    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset+'_'+str(args.input_size),
        str(args.epochs)+'_'+str(args.mask_ratio)+'_'+str(args.blr)+'_'+str(args.weight_decay)+'_'+str(args.batch_size*args.device_num)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(dataset_train)

    if 'mae_vit_' in args.model:
        print('MAE pretraining ViT series model')
        model = models_mae.__dict__[args.model](mask_ratio=args.mask_ratio, norm_pix_loss=args.norm_pix_loss)
    elif 'mae_vitae_' in args.model:
        # Todo: Finish the mae_vitae model
        print('MAE pretraining ViTAE series model')
        # model = models_mae_vitae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        raise NotImplementedError
    else:
        raise NotImplementedError

    # print("Model = %s" % str(model))

    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    eff_batch_size = args.batch_size * args.accum_iter * args.device_num

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  # 累积iter, lr会增加

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # define lr_schedule
    lr_schedule = Tensor(adjust_learning_rate(new_epochs, args))

    # define optimizer
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    group_params = add_weight_decay(model, args.weight_decay)
    optimizer = nn.AdamWeightDecay(group_params,
                                   learning_rate=lr_schedule,
                                   beta1=args.beta1,
                                   beta2=args.beta2,
                                   eps=1e-08,
                                   weight_decay=0.01)
    print(optimizer)

    # load pretrain ckpt
    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if args.use_ckpt:
        params_dict = load_checkpoint(args.use_ckpt)
        load_param_into_net(model, params_dict)
        load_param_into_net(optimizer, params_dict)

    # define model
    # from util.misc import NativeScalerWithGradNormCount as NativeScaler
    # loss_scaler = NativeScaler()
    train_model = create_train_one_step(args, model, optimizer, logger)

    # define callback
    """
    The old API of LossMonitor only support 'per_print_times' parameter.
    In new version of LuoJiaNET, we support the log input.
    callback = [LossMonitor(per_print_times=args.per_step_size, log=logger),]
    """
    callback = [LossMonitor(per_print_times=args.per_step_size), ]

    # define ckpt config
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq,
                                     keep_checkpoint_max=args.keep_checkpoint_max,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix,
                                     directory=args.output_dir,
                                     config=config_ck)
        callback += [ckpoint_cb, ]

    # define model and begin training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    model = Model(train_model)
    model.train(epoch=new_epochs,
                train_dataset=dataset_train,
                callbacks=callback,
                dataset_sink_mode=(args.device_target != "CPU"),
                sink_size=args.per_step_size)  # 'sink_size' is invalid if 'dataset_sink_mode' is False.
    # model.train(epoch=new_epochs,
    #             train_dataset=dataset_train,
    #             callbacks=callback,
    #             dataset_sink_mode=(args.device_target != 'CPU'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)