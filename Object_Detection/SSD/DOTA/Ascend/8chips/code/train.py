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

"""Train SSD and get checkpoint files."""

import argparse
import ast
import os
import luojianet_ms.nn as nn
from luojianet_ms import context, Tensor
from luojianet_ms.communication.management import init, get_rank
from luojianet_ms.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from luojianet_ms.train import Model
from luojianet_ms.context import ParallelMode
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, dtype
from src.ssd import SSDWithLossCell, TrainingWrapper, ssd_resnet34
from src.config import config
from src.dataset import create_ssd_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter_by_list


set_seed(1)


def get_args():
    """
    Get args
    """
    parser = argparse.ArgumentParser(description="SSD_ResNet34 training")
    parser.add_argument("--data_url", type=str, default="/media/xx/新加卷/DATA/NWPU")
    parser.add_argument("--train_url", type=str, default="/media/xx/新加卷/OUTPUT/NWPU/ssd/test2/train")
    parser.add_argument("--mindrecord_url", type=str, default="/media/xx/新加卷/DATA/NWPU/NWPU_SSD_TRAIN")
    parser.add_argument("--run_platform", type=str, default="GPU", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend, GPU and CPU.")
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=1, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--epoch_size", type=int, default=500, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    args_opt = parser.parse_args()
    return args_opt

def ssd_model_build(args_opt):
    """
    Build ssd_resnet34.
    """
    if config.model == "ssd_resnet34":
        ssd = ssd_resnet34(config=config)
        init_net_param(ssd)
        if config.feature_extractor_base_param != "":
            param_dict = load_checkpoint(config.feature_extractor_base_param)
            for x in list(param_dict.keys()):
                param_dict["network.feature_extractor.resnet." + x] = param_dict[x]
                del param_dict[x]
            load_param_into_net(ssd.feature_extractor.resnet, param_dict)
    else:
        raise ValueError(f'config.model: {config.model} is not supported')
    return ssd


def main():
    """
    Execute training process!
    """
    args_opt = get_args()
    rank = 0
    device_num = 1
    config.coco_root = args_opt.data_url
    config.mindrecord_dir = args_opt.mindrecord_url
    local_train_url = args_opt.train_url

    if args_opt.run_platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform)

    if args_opt.distribute:
        init()
        device_num = int(os.getenv("RANK_SIZE"))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
        rank = get_rank()

    mindrecord_file = create_mindrecord(args_opt.dataset, "ssd.mindrecord", True)

    if args_opt.only_create_dataset:
        return

    loss_scale = float(args_opt.loss_scale)
    if args_opt.run_platform == "CPU":
        loss_scale = 1.0

    # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
    use_multiprocessing = (args_opt.run_platform != "CPU")
    dataset = create_ssd_dataset(mindrecord_file, repeat_num=1, batch_size=args_opt.batch_size,
                                 device_num=device_num, rank=rank, use_multiprocessing=use_multiprocessing)
    dataset_size = dataset.get_dataset_size()
    print(f"Create dataset done! dataset size is {dataset_size}")
    ssd = ssd_model_build(args_opt)

    if ("use_float16" in config and config.use_float16) or args_opt.run_platform == "GPU":
        ssd.to_float(dtype.float16)
    net = SSDWithLossCell(ssd, config)

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs,
                                   keep_checkpoint_max=400)
    ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=local_train_url + "/card{}".format(rank), config=ckpt_config)

    if args_opt.pre_trained:
        local_pre_train = args_opt.pre_trained
        param_dict = load_checkpoint(local_pre_train)
        if args_opt.filter_weight:
            filter_checkpoint_parameter_by_list(param_dict, config.checkpoint_filter_list)
        load_param_into_net(net, param_dict, True)

    lr = Tensor(get_lr(global_step=args_opt.pre_trained_epoch_size * dataset_size,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * args_opt.lr, lr_max=args_opt.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size))

    if "use_global_norm" in config and config.use_global_norm:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, 1.0)
        net = TrainingWrapper(net, opt, loss_scale, True)
    else:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]
    model = Model(net)

    dataset_sink_mode = False
    if args_opt.mode == "sink" and args_opt.run_platform != "CPU":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)

if __name__ == '__main__':
    main()
