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
import importlib
import os
import shutil
from utils.deeplearning_dp import train_net
from utils.random_seed import setup_seed
from dataset.isprs_dataset import Isprs_Dataset
import dataset.isprs_transform as transform
from model import SegModel
from luojianet_ms import context
import moxing as mox


def get_argparse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-c', '--config', type=str, default='Deepv3', help='Configuration File')
    return parser.parse_args()

config_name=get_argparse().config
param = importlib.import_module("." + get_argparse().config, package='config').param

context.set_context(mode=context.PYNATIVE_MODE,device_id=param['device_id'], device_target=param['device_target'])



if __name__ == "__main__":
    # load data from obs
    mox.file.copy_parallel('obs://luojianet-benchmark-dataset/Semantic_Segmentation/Vaihingen_split/', param['data_dir'])
    

    # set random seed
    config_name=get_argparse().config
    param = importlib.import_module("." + get_argparse().config, package='config').param
    setup_seed(param['random_seed'])

    # data path
    data_dir = param['data_dir']#'./data'
    train_img_dir_path = os.path.join(data_dir, "cropped512/img")
    train_label_dir_path = os.path.join(data_dir, "cropped512/label")
    val_img_dir_path = os.path.join(data_dir, "cropped512/img")
    val_label_dir_path = os.path.join(data_dir, "cropped512/label")
    train_image_id_txt_path = param['train_image_id_txt_path']
    val_image_id_txt_path = param['val_image_id_txt_path']


    # dataset
    train_transform=getattr(transform,param['train_transform'])()
    val_transform = getattr(transform, param['val_transform'])()
    train_dataset = Isprs_Dataset(img_dir=train_img_dir_path, label_dir=train_label_dir_path,
                                      img_id_txt_path=train_image_id_txt_path, transform=train_transform)
    valid_dataset = Isprs_Dataset(img_dir=val_img_dir_path, label_dir=val_label_dir_path,img_id_txt_path=val_image_id_txt_path,
                                      transform=val_transform)

    # model
    model = SegModel(model_network=param['model_network'],
                     in_channels=param['in_channels'], n_class=param['n_class'])

    # model save path
    save_ckpt_dir = os.path.join('/cache/checkpoint', param['save_dir'], 'ckpt')
    save_log_dir = os.path.join('/cache/checkpoint', param['save_dir'])
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    param['save_log_dir'] = save_log_dir
    old_config_name_path='/home/ma-user/modelarts/user-job-dir/code/config'+'/'+config_name+'.py'
    new_config_name_path = param['save_log_dir'] + '/' + config_name + '.py'
    shutil.copyfile(src=old_config_name_path,dst=new_config_name_path)
    param['save_ckpt_dir'] = save_ckpt_dir

    # training
    train_net(param=param, model=model, train_dataset=train_dataset, valid_dataset=valid_dataset)

    # upload opt to obs
    mox.file.copy_parallel('/cache/checkpoint', 'obs://luojianet-benchmark/Semantic_Segmentation/DeepLabv3/ISPRS_Vaihingen/Ascend/1chip/ckpt/')
