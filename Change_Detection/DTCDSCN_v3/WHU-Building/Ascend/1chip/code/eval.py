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

from turtle import forward
from module import CDNet34
import luojianet_ms
import numpy as np
from PIL import Image
import cv2
from luojianet_ms import Tensor, load_checkpoint, load_param_into_net
import os
import tqdm
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision
from luojianet_ms import context
import luojianet_ms.ops as ops
from luojianet_ms import context
from luojianet_ms.common import set_seed
from luojianet_ms import load_checkpoint, load_param_into_net
from dataset import create_Dataset
from config import config
import argparse
import moxing as mox


# caculate precision between output and target
def precision(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + smooth)


# caculate recall between output and target
def recall(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (intersection + smooth) / (target.sum() + smooth)


# caculate F1 score between output and target
def F1_score(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


# caculate IoU between output and target
def IoU(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + target.sum() - intersection + smooth)


# caculate Kappa score between output and target
def Kappa(output, target):
    smooth = 1e-5
    TP = (output * target).sum()  # TP
    TN = ((1 - output) * (1 - target)).sum()  # TN
    FP = (output * (1 - target)).sum()  # FP
    FN = ((1 - output) * target).sum()  # FN
    n = TP + TN + FP + FN
    p0 = (TP + TN + smooth) / (n + smooth)
    a1 = TP + FP
    a2 = FN + TN
    b1 = TP + FN
    b2 = FP + TN
    pe = (a1 * b1 + a2 * b2 + smooth) / (n * n + smooth)
    return (p0 - pe + smooth) / (1 - pe + smooth)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def infer(model_path, data_path):
    '''inference_dataset'''
    model = CDNet34(3)
    model.set_train(False)
    print('load test weights from %s', str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    print('loaded test weights from %s', str(model_path))
    val_dataset, _ = create_Dataset(data_path, aug = False, batch_size = 1, shuffle = False)
    data_loader = val_dataset.create_dict_iterator()
    precisions = AverageMeter()
    recalls = AverageMeter()
    F1_scores = AverageMeter()
    IoUs = AverageMeter()
    Kappas = AverageMeter()
    for _, data in enumerate(data_loader):
        output = model(data["image"])[2].asnumpy()
        output = ((output > 0.484) * 255).astype('uint8')
        output = output / 255
        pre = precision(output, data["mask"].asnumpy())
        precisions.update(pre, 1)
        Recall = recall(output, data["mask"].asnumpy())
        recalls.update(Recall, 1)
        f1 = F1_score(output, data["mask"].asnumpy())
        F1_scores.update(f1, 1)
        iou = IoU(output, data["mask"].asnumpy())
        IoUs.update(iou, 1)
        kappa = Kappa(output, data["mask"].asnumpy())
        Kappas.update(kappa, 1)
    print("Final precisions: " + str(precisions.avg))
    print("Final recalls: " + str(recalls.avg))
    print("Final F1 scores: " + str(F1_scores.avg))
    print("Final IoUs: " + str(IoUs.avg))
    print("Final Kappas: " + str(Kappas.avg))


if __name__ == '__main__':
    ckpt_obs = 'obs://luojianet-benchmark/Change_Detection/DTCDSCN_v3/WHU-BCD/1chip/code/best.ckpt'
    ckpt_obs =  'obs://luojianet-benchmark/Change_Detection/DTCDSCN_v3/WHU-BCD/1chip/ckpt/model/DTCDSCN-200_336.ckpt'
    ckpt_cache = '/cache/ckpt/test.ckpt'
    mox.file.copy_parallel(ckpt_obs, ckpt_cache)
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Change_Detection/WHU_CD_data_split/A_test/', config.dataset_path+'A/')
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Change_Detection/WHU_CD_data_split/B_test/', config.dataset_path+'B/')
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Change_Detection/WHU_CD_data_split/building_A_test/', config.dataset_path+'building_A/')
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Change_Detection/WHU_CD_data_split/building_B_test/', config.dataset_path+'building_B/')
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Change_Detection/WHU_CD_data_split/label_test/', config.dataset_path+'label/')


    parser = argparse.ArgumentParser(description = 'Change Detection')
    parser.add_argument('--checkpoint_path', type = str, default = ckpt_cache, help = 'Saved checkpoint file path')
    parser.add_argument('--dataset_path', type = str, default = config.dataset_path, help = 'Eval dataset path')
    parser.add_argument('--device_target', type = str, default = config.device_target, help = 'Device target')
    parser.add_argument('--device_id', type = int, default = config.device_id, help = 'Device id')
    args = parser.parse_args()
    set_seed(1)

    context.set_context(device_target = args.device_target, device_id = args.device_id)
    if (args.checkpoint_path and args.dataset_path):
        infer(args.checkpoint_path, args.dataset_path)
    else:
        print("Error:There are no images to predict or no weights!")
