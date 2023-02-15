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

"""Config parameters for SSD_ResNet34 models."""

from easydict import EasyDict as ed

config = ed({
    "model": "ssd_resnet34",
    "img_shape": [300, 300],
    "num_ssd_boxes": 8732,
    "match_threshold": 0.5,
    "nms_threshold": 0.7,
    "min_score": 0.4,
    "max_boxes": 320,

    "global_step": 0,
    "lr_init": 0.0025,
    "lr_end_rate": 0.001,
    "warmup_epochs": 2,
    "weight_decay": 4e-5,
    "momentum": 0.9,

    # network
    "num_default": [4, 6, 6, 6, 4, 4],
    "extras_in_channels": [],
    "extras_out_channels": [256, 512, 512, 256, 256, 256],
    "extras_strides": [1, 1, 2, 2, 2, 1],
    "extras_ratios": [0.2, 0.2, 0.2, 0.25, 0.5, 0.25],
    "feature_size": [38, 19, 10, 5, 3, 1],
    "min_scale": 0.2,
    "max_scale": 0.95,
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "steps": (7, 15, 30, 60, 100, 300),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,  # modify
    "alpha": 0.75,  # modify

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    "feature_extractor_base_param": "",
    "checkpoint_filter_list": ['multi_loc_layers', 'multi_cls_layers'],
    "mindrecord_dir": "/media/xx/新加卷/DATA/NWPU/NWPU_SSD_TRAIN",
    "coco_root": "/media/xx/新加卷/DATA/NWPU",
    "train_data_type": "train2017",
    "val_data_type": "val2017",
    "instances_set": "annotations/instances_{}.json",
    "classes": ('background','airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
          'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle'),
    "num_classes": 11,
    # The annotation.json position of voc validation dataset.
    "voc_json": "annotations/voc_instances_val.json",
    # voc original dataset.
    "voc_root": "/data/voc_dataset",
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    "image_dir": "/media/xx/新加卷/DATA/NWPU/train2017",
    "anno_path": "/media/xx/新加卷/DATA/NWPU/nwpu_train.txt",
    "outputs_dir": "/media/xx/新加卷/OUTPUT/NWPU/ssd/test3/train",
    # infer options
    "infer_log": "/home/xx/Desktop/ssd_resnet34/infer_result/",
    # device
    "rank": 0,
})
