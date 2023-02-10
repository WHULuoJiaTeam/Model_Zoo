import os
import ast
import argparse
from unittest import result
import pandas as pd
from luojianet_ms import context, nn
from luojianet_ms.train.model import Model
from luojianet_ms.common import set_seed
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net

from config import *
from utils import create_dataset,CrossEntropySmooth
from vgg import *
# from Resnet import *
# from Resnet_se import *

set_seed(1)
CACHE = "/cache/data/"
CKPT_CACHE = "/cache/ckpt/"
import moxing as mox

if __name__ == '__main__':
    ckpt_obs = 'obs://luojianet-benchmark/Scene_Classification/VGG-16/RSSCN7/8chip/ckpt/model/net-346_25.ckpt'
    ckpt_cache = '/cache/ckpt/test.ckpt'
    mox.file.copy_parallel(ckpt_obs, ckpt_cache)
    datasetpath = '/cache/data'
    mox.file.copy_parallel(config.dataset_path , datasetpath )

    parser = argparse.ArgumentParser(description='Image classification')

    parser.add_argument('--dataset_path', type=str, default=datasetpath, help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default=ckpt_cache, help='Saved checkpoint file path')
    parser.add_argument('--device_target', type=str, default="Ascend", help='Device target')

    args_opt = parser.parse_args()
    context.set_context(device_target=args_opt.device_target)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                        do_train=False,
                        batch_size=config.batch_size)
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    step_size = dataset.get_dataset_size()

    # define net
    net = vgg16_bn(num_classes=config.class_num)

    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define model
    eval_metrics = {'Loss': nn.Loss(),
                    'Top_1_Acc': nn.Top1CategoricalAccuracy(),
                    'Top_5_Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss_fn=loss, metrics=eval_metrics)

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)

