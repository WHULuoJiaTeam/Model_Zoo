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

# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""YoloV3 train."""
# try:
#     import luojianet_ms
# except:
import os
import time
import datetime
import argparse

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.communication as comm
from luojianet_ms.train.callback import ModelCheckpoint, RunContext, CheckpointConfig
from luojianet_ms import Tensor

from src.yolo import YOLOV3DarkNet53, YoloWithLossCell
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups, cpu_affinity
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov3_params
from src.util import keep_loss_fp32, EvalCallBack, apply_eval

from model_utils.config import config

# only useful for huawei cloud modelarts.
from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process

ms.set_seed(1)


def get_args():
    parser = argparse.ArgumentParser(description="Yolov3 training")
    parser.add_argument("--use_modelarts", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/data02/tjy/DOTA-V1.5")
    parser.add_argument("--per_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--T_max", type=int, default=3)
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--warmup_epochs", type=int, default=4)
    parser.add_argument("--ckpt_path", type=str, default="/home/ma-user/work/OUTPUT/yolov4/mult/test1")
    parser.add_argument("--is_distributed", type=int, default=1)
    parser.add_argument("--ckpt_max_num", type=int, default=280)
    parser.add_argument("--max_box", type=int, default=160)
    parser.add_argument("--num_classes", type=int, default=10)
    args_opt = parser.parse_args()
    return args_opt

class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

def conver_training_shape(args):
    training_shape = [int(args.training_shape), int(args.training_shape)]
    return training_shape


def set_graph_kernel_context():
    if ms.get_context("device_target") == "GPU":
        ms.set_context(enable_graph_kernel=True)
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion "
                                          "--enable_trans_op_optimize "
                                          "--disable_cluster_ops=ReduceMax,Reshape "
                                          "--enable_expand_ops=Conv2D")


def network_init(args):
    device_id = int(os.getenv('DEVICE_ID', '0'))
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, save_graphs=False, device_id=device_id)
    # set_graph_kernel_context()

    # Set mempool block size for improving memory utilization, which will not take effect in GRAPH_MODE
    if ms.get_context("mode") == ms.PYNATIVE_MODE:
        ms.set_context(mempool_block_size="31GB")
        # Since the default max memory pool available size on ascend is 30GB,
        # which does not meet the requirements and needs to be adjusted larger.
        if ms.get_context("device_target") == "Ascend":
            ms.set_context(max_device_memory="31GB")

    profiler = None
    if args.need_profiler:
        profiling_dir = os.path.join("profiling",
                                     datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        profiler = ms.profiler.Profiler(output_path=profiling_dir)

    # init distributed
    if args.is_distributed:
        comm.init()
        args.rank = comm.get_rank()
        args.group_size = comm.get_group_size()
        if args.device_target == "GPU" and args.bind_cpu:
            cpu_affinity(args.rank, min(args.group_size, args.device_num))

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1
    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)
    args.logger.save_args(args)
    # args.loss_log = get_logger(args.loss_path, args.rank)
    # args.epochtime_log = get_logger(args.epochtime_path, args.rank)
    return profiler


def parallel_init(args):
    ms.reset_auto_parallel_context()
    parallel_mode = ms.ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ms.ParallelMode.DATA_PARALLEL
        degree = comm.get_group_size()
    ms.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    args = get_args()
    config.T_max = args.T_max
    config.max_epoch = args.max_epoch
    """Train function."""
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.T_max:
        config.T_max = config.max_epoch
    config.use_modelarts = args.use_modelarts
    config.data_dir = args.data_dir
    config.per_batch_size = args.per_batch_size
    config.lr = args.lr
    config.warmup_epochs = args.warmup_epochs
    config.ckpt_path = args.ckpt_path
    config.is_distributed = args.is_distributed
    config.ckpt_max_num = args.ckpt_max_num
    config.max_box = args.max_box
    config.num_classes = args.num_classes
    
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.data_root = os.path.join(config.data_dir, 'train2017')
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_train2017.json')
    config.data_val_root = os.path.join(config.data_dir, 'val2017')
    config.ann_val_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    profiler = network_init(config)

    loss_meter = AverageMeter('loss')
    parallel_init(config)

    network = YOLOV3DarkNet53(is_training=True)
    if config.run_eval:
        network_eval = network
        # keep_eval_fp32(network)
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov3_params(config, network)

    network = YoloWithLossCell(network)
    config.logger.info('finish get network')

    if config.training_shape:
        config.multi_scale = [conver_training_shape(config)]

    ds, ds_size = create_yolo_dataset(image_dir=config.data_root, anno_path=config.annFile, is_training=True,
                             batch_size=config.per_batch_size, device_num=config.group_size,
                             rank=config.rank, config=config)
    config.logger.info('Finish loading dataset')


    config.steps_per_epoch = ds.get_dataset_size()

    if config.ckpt_interval <= 0:
        config.ckpt_interval = config.steps_per_epoch

    lr = get_lr(config)
    opt = nn.Momentum(params=get_param_groups(network), momentum=config.momentum, learning_rate=ms.Tensor(lr),
                      weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    is_gpu = ms.get_context("device_target") == "GPU"
    if is_gpu:
        loss_scale_value = 1.0
        loss_scale = ms.FixedLossScaleManager(loss_scale_value, drop_overflow_update=False)
        network = ms.build_train_network(network, optimizer=opt, loss_scale_manager=loss_scale,
                                         level="O2", keep_batchnorm_fp32=False)
        keep_loss_fp32(network)
    else:
        network = nn.TrainOneStepCell(network, opt, sens=config.loss_scale)
        network.set_train()

    network.set_train(True)

    if config.rank_save_ckpt_flag or config.run_eval:
        cb_params = InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)

    if config.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=config.ckpt_max_num)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(config.rank))
        ckpt_cb.begin(run_context)

    if config.run_eval:
        data_val_root = config.data_val_root
        ann_val_file = config.ann_val_file
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        input_val_shape = Tensor(tuple(config.test_img_shape), ms.float32)
        # init detection engine
        eval_dataset, eval_data_size = create_yolo_dataset(data_val_root, ann_val_file, is_training=False,
                                                           batch_size=config.per_batch_size, device_num=1,
                                                           rank=0, shuffle=False, config=config)
        eval_param_dict = {"net": network_eval, "dataset": eval_dataset, "data_size": eval_data_size,
                           "anno_json": ann_val_file, "input_shape": input_val_shape, "args": config}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=save_ckpt_path, besk_ckpt_name="best_map.ckpt",
                               metrics_name="mAP")




    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True)
    first_step = True
    stop_profiler = False

    config.log_interval = config.steps_per_epoch
    for epoch_idx in range(config.max_epoch):
        epoch_start_time = time.time()
        for step_idx, data in enumerate(data_loader):
            images = data["image"]
            # input_shape = images.shape[2:4]
            # config.logger.info('iter[{}], shape{}'.format(step_idx, input_shape[0]))
            images = ms.Tensor.from_numpy(images)

            batch_y_true_0 = ms.Tensor.from_numpy(data['bbox1'])
            batch_y_true_1 = ms.Tensor.from_numpy(data['bbox2'])
            batch_y_true_2 = ms.Tensor.from_numpy(data['bbox3'])
            batch_gt_box0 = ms.Tensor.from_numpy(data['gt_box1'])
            batch_gt_box1 = ms.Tensor.from_numpy(data['gt_box2'])
            batch_gt_box2 = ms.Tensor.from_numpy(data['gt_box3'])

            loss = network(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                           batch_gt_box2)
            loss_meter.update(loss.asnumpy())

            # it is used for loss, performance output per config.log_interval steps.
            if (epoch_idx * config.steps_per_epoch + step_idx) % config.log_interval == 0:
                time_used = time.time() - t_end
                if first_step:
                    fps = config.per_batch_size * config.group_size / time_used
                    per_step_time = time_used * 1000
                    first_step = False
                else:
                    fps = config.per_batch_size * config.log_interval * config.group_size / time_used
                    per_step_time = time_used / config.log_interval * 1000
                config.logger.info('epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, '
                                   'lr:{}, per step time: {}ms'.format(epoch_idx + 1, step_idx + 1,
                                                                       loss_meter, fps, lr[step_idx], per_step_time))
                t_end = time.time()
                loss_meter.reset()
            if config.need_profiler:
                if epoch_idx * config.steps_per_epoch + step_idx == 10:
                    profiler.analyse()
                    stop_profiler = True
                    break
        epoch_end_time = time.time()
        epoch_one_time = (epoch_end_time - epoch_start_time) * 1000
        # config.epochtime_log.info('epoch[{}], time:{}ms'.format(epoch_idx + 1, epoch_one_time))
        # config.loss_log.info('epoch:[{}], {}'.format(epoch_idx + 1, loss_meter))
        if config.rank_save_ckpt_flag:
            ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank))
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path, exist_ok=True)
            ckpt_name = os.path.join(ckpt_path, "yolov3_{}_{}.ckpt".format(epoch_idx + 1, config.steps_per_epoch))
            ms.save_checkpoint(network, ckpt_name)
            if config.run_eval:
                eval_cb.epoch_end(run_context)
                network.set_train()
            cb_params.cur_epoch_num += 1
        if stop_profiler:
            break

    config.logger.info('==========end training===============')


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1, 7'
    run_train()
