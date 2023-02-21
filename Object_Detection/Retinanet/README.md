# Contents

- [Contents](#contents)
  - [Retinanet Description](#retinanet-description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [CKPT](#ckpt)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
      - [Usage](#usage)
      - [Run](#run)
      - [Result](#result)
    - [Evaluation Process](#evaluation-process)
      - [Usage](#usage)
      - [Run](#run)
      - [Result](#result)
    - [Export MindIR](#export-mindir)
      - [Usage](#usage)
      - [Run](#run)
    - [Inference Process](#inference-process)
      - [Usage](#usage)
  - [Model Description](#model-description)
    - [Performance](#performance)
      - [Train Performance](#train-performance)
      - [Evaluation Performance](#evaluation-performance)
  - [Description of Random Situation](#description-of-random-situation)

# Retinanet Description

The RetinaNet algorithm was derived from a 2018 Facebook AI Research paper Focal Loss for Dense Object Detection. The biggest contribution of this paper is that Focal Loss is proposed to solve the problem of category imbalance, thus creating the RetinaNet (one-stage target detection algorithm), a target detection network whose accuracy exceeds that of classic two-stage Faster-RCNN.

[Paper](https://arxiv.org/pdf/1708.02002.pdf)

Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017:2999-3007.

# Model Architecture

The architecture of Retinanet :

[链接](https://arxiv.org/pdf/1708.02002.pdf)

# Dataset

Dataset used:  [DOTA-V1.5]([DOTA (captain-whu.github.io)](https://captain-whu.github.io/DOTA/dataset.html)). The image is clipped to 600*600 pixel size and 20% overlap. You can get the process code and processed dataset form here.

* Process code: https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/preprocess.zip
* Dataset（clipped）: https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/DOTA.zip

Dataset support: [COCO2017] or datasetd with the same format as MS COCO
Annotation support: [COCO2017] or annotation as the same format as MS COCO

- The directory structure is as follows, the name of directory and file is user define:

  ```text
      ├── dataset
          ├── DOTA(coco_root)
              ├── annotations
              │   ├─ train.json
              │   └─ val.json
              ├─train
              │   ├─picture1.jpg
              │   ├─ ...
              │   └─picturen.jpg
              ├─ val
                  ├─picture1.jpg
                  ├─ ...
                  └─picturen.jpg
  ```

we suggest user to use MS COCO dataset to experience our model,
other datasets need to use the same format as MS COCO.

# Environment Requirements

- Hardware（Ascend/GPU）
- Prepare hardware environment with Ascend or GPU processor.
- Framework
  - [Luojianet]([首页](http://58.48.42.237/luojiaNet/home))
- For more information, please check the resources below：
  - [Luojianet Tutorials]([初学入门](http://58.48.42.237/luojiaNet/tutorial/quickstart))
  - [Luojianet API]([API](http://58.48.42.237/luojiaNet/luojiaNetapi/))

# CKPT

Here we give the .ckpt file. You can use this file to pretrain, eval and infer. The link is as follows:

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/RetinaNet/DOTA/8chips/test5_fine_test4/train/retinanet-100_94.ckpt

# Script Description

## Script and Sample Code

```text
.
└─Retinanet
  ├─README.md
  ├─ascend310_infer                           # launch inference in ascend
  ├─scripts
    ├─run_single_train.sh                     # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh                 # launch distributed training(8p) in ascend
    ├─run_distribute_train_gpu.sh             # launch distributed training(8p) in gpu
    ├─run_single_train_gpu.sh                 # launch standalone training(1p) in gpu
    ├─run_infer_310.sh                        # launch inference in ascend
    ├─run_eval.sh                             # launch evaluating in ascend
    ├─run_eval_gpu.sh                         # launch evaluating in gpu
  ├─config
    ├─finetune_config.yaml                      # parameter configuration
    └─default_config.yaml                       # parameter configuration
  ├─src
    ├─dataset.py                              # Data preprocessing
    ├─retinanet.py                            # Network
    ├─init_params.py                          # initializer of parameters
    ├─lr_generator.py                         # generate learning rate
    ├─coco_eval                               # coco eval
    ├─box_utils.py                            # box util function
    ├─_init_.py                               # initialize
    ├──model_utils
      ├──config.py                            # parameter configuration
      ├──device_adapter.py                    # device info
      ├──local_adapter.py                     # device info
      ├──moxing_adapter.py                    # decorator(for ModelArts data copying)
  ├─train.py                                  # train network
  ├─export.py                                 # export AIR,MINDIR model
  ├─postprogress.py                           # postprogress in Ascend310
  └─eval.py                                   # eval
  └─create_data.py                            # create mindrecord dataset
  └─data_split.py                             # split data for transfer
  └─quick_start.py                            # Transfer learning visualization
  └─default_config.yaml                       # parameter configuration
```



### Script Parameters

```text
在脚本中使用到的主要参数是：
"img_shape": [600, 600],                                                                        # image shape
"num_retinanet_boxes": 67995,                                                                   # anchors
"match_thershold": 0.5,                                                                         # match thershold
"nms_thershold": 0.6,                                                                           # nms thershold
"min_score": 0.1,                                                                               # Minimum score
"max_boxes": 100,                                                                               # Maximum number of detection boxes
"lr_init": 1e-6,                                                                                # Initial learning rate
"lr_end_rate": 5e-3,                                                                            # Ratio of final learning rate to maximum learning rate
"warmup_epochs1": 2,                                                                            # Number of warmup cycles in the first stage
"warmup_epochs2": 5,                                                                            # Number of warmup cycles in the second stage
"warmup_epochs3": 23,                                                                           # Number of warmup cycles in the third stage
"warmup_epochs4": 60,                                                                           # Number of warmup cycles in the fourth stage
"warmup_epochs5": 160,                                                                          # Number of warmup cycles in the fifth stage
"momentum": 0.9,                                                                                # momentum
"weight_decay": 1.5e-4,                                                                         # Weight decay rate
"num_default": [9, 9, 9, 9, 9],                                                                 # The number of prior boxes in a single grid
"extras_out_channels": [256, 256, 256, 256, 256],                                               # Number of output channels at the feature layer
"feature_size": [75, 38, 19, 10, 5],                                                            # Feature layer size
"aspect_ratios": [[0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0], [0.5,1.0,2.0]],   # anchors size change ratio
"steps": [8, 16, 32, 64, 128],                                                                 # anchors stride
"anchor_size":[32, 64, 128, 256, 512],                                                          # anchors shape
"prior_scaling": [0.1, 0.2],                                                                    # Used to adjust the ratio of regression to regression in loss
"gamma": 2.0,                                                                                   # parameter in focal loss
"alpha": 0.75,                                                                                  # parameter in focal loss
"mindrecord_dir": "/cache/MindRecord_COCO",                                                     # mindrecord file path
"coco_root": "/cache/coco",                                                                     # coco dataset path
"train_data_type": "train2017",                                                                 # train floder name
"val_data_type": "val2017",                                                                     # val floder name
"instances_set": "annotations_trainval2017/annotations/instances_{}.json",                      # annotations path
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',     # coco数据集的种类
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush'),
"num_classes": 81,                                                                              # Number of categories
"voc_root": "",                                                                                 # voc data path
"voc_dir": "",
"image_dir": "",                                                                                # image path
"anno_path": "",                                                                                # annotations path
"save_checkpoint": True,                                                                        # save checkpoint
"save_checkpoint_epochs": 1,                                                                    # Numbers of checkpoint epoch
"keep_checkpoint_max":1,                                                                        # Maximum of saved checkpoint
"save_checkpoint_path": "./ckpt",                                                              # saved checkpoint path
"finish_epoch":0,                                                                               # finished epochs
"checkpoint_path":"/home/hitwh1/1.0/ckpt_0/retinanet-500_458_59.ckpt"                           # checkpoint path for evluation
```

### Training Process

#### Usage

Use shell scripts for training. The shell script is used as follows:

```shell
# 8p

创建 RANK_TABLE_FILE
bash scripts/run_distribute_train.sh DEVICE_NUM RANK_TABLE_FILE CONFIG_PATH MINDRECORD_DIR PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

# 1p

bash scripts/run_single_train.sh DEVICE_ID MINDRECORD_DIR CONFIG_PATH PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)
```

RANK_TABLE_FILE related resources see [link](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html), Get device_ip method see [link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。

#### Run

```text
数据集结构
└─cocodataset
  ├─train2017
  ├─val2017
  ├─test2017
  ├─annotations

```

```default_config.yaml
Before training, create MindRecord file first. Take COCO data set as an example. yaml file configures the path of coco data set and mindrecord storage path
# your cocodataset dir
coco_root: /home/DataSet/cocodataset/
# mindrecord dataset dir
mindrecord_dr: /home/DataSet/MindRecord_COCO
```

```MindRecord
# Generate training data sets 
python create_data.py --create_dataset coco --prefix retinanet.mindrecord --is_training True --config_path
(eg：python create_data.py  --create_dataset coco --prefix retinanet.mindrecord --is_training True --config_path /home/retinanet/config/default_config.yaml)
# Generate evaluation data sets
python create_data.py --create_dataset coco --prefix retinanet_eval.mindrecord --is_training False --config_path
(eg：python create_data.py  --create_dataset coco --prefix retinanet.mindrecord --is_training False --config_path /home/retinanet/config/default_config.yaml)
```

```bash
Ascend:
# 8p：
bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [MINDRECORD_DIR] [CONFIG_PATH] [PRE_TRAINED(optional)] [PRE_TRAINED_EPOCH_SIZE(optional)]
# example: bash scripts/run_distribute_train.sh 8 ~/hccl_8p.json /home/DataSet/MindRecord_COCO/ /home/retinanet/config/default_config.yaml

# 1p：
bash scripts/run_single_train.sh [DEVICE_ID] [MINDRECORD_DIR] [CONFIG_PATH]
# example: bash scripts/run_single_train.sh 0 /home/DataSet/MindRecord_COCO/ /home/retinanet/config/default_config.yaml
```

```bash
GPU:
# 8p：
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [MINDRECORD_DIR] [CONFIG_PATH] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [PRE_TRAINED(optional)] [PRE_TRAINED_EPOCH_SIZE(optional)]
# example: bash scripts/run_distribute_train_gpu.sh 8 /home/DataSet/MindRecord_COCO/ /home/retinanet/config/default_config_gpu.yaml 0,1,2,3,4,5,6,7
```

#### Result

Training results are stored in the sample path. checkpoint is stored in the './ckpt 'path, and the training logs are stored in the'./log.txt 'path.

```text
# 8p:
epoch: 1 step: 1, loss is 0.5481206178665161
lr:[0.000032]
epoch: 1 step: 1, loss is 0.6371729373931885
lr:[0.000032]
epoch: 1 step: 1, loss is 0.568812370300293epoch: 1 step: 1, loss is 0.5394906401634216
epoch: 1 step: 1, loss is 0.6255771517753601lr:[0.000032]
lr:[0.000032]
lr:[0.000032]
epoch: 1 step: 1, loss is 0.5361188650131226
lr:[0.000032]
epoch: 1 step: 1, loss is 0.47298291325569153
lr:[0.000032]
epoch: 1 step: 1, loss is 0.48077914118766785
lr:[0.000032]
epoch: 1 step: 2, loss is 0.5254751443862915
lr:[0.000064]
```

### Evaluation Process

#### Usage

Use shell scripts for evaluation. The shell script is used as follows:

```bash
Ascend:
bash scripts/run_eval.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH] [CONFIG_PATH]
# example: bash scripts/run_eval.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/retinanet/ckpt/retinanet_500-458.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json /home/retinanet/config/default_config.yaml
```

```bash
GPU:
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH] [CONFIG_PATH]
# example: bash scripts/run_eval_gpu.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/retinanet/ckpt/retinanet_500-458.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json /home/retinanet/config/default_config_gpu.yaml
```

#### Result

The results are stored in the sample path, which you can view in 'eval.log'.

```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.317
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.147
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.369

========================================
mAP: 0.3016772742145538
```

### Export MindIR

#### Usage

Before exporting the model, change the checkpoint_path configuration item in the config.py file. The value is the checkpoint path.

```shell
python export.py --file_name [RUN_PLATFORM] --file_format[EXPORT_FORMAT] --checkpoint_path [CHECKPOINT PATH]
```

`EXPORT_FORMAT` options:  ["AIR", "MINDIR"]

#### Run

```shell
python export.py  --file_name retinanet --file_format MINDIR --checkpoint_path /cache/checkpoint/retinanet_550-458.ckpt
```

### Inference Process

#### Usage

Method 1. Use inference.py

```shell
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

Method 2. Use .sh

Before inference, it is necessary to complete the derivation of the model on Senteng 910 environment. Exclude images with iscrowd as true when reasoning. The excluded image id is saved in the ascend310_infer directory.

In addition, configuration items coco_root, val_data_type and instances_set in the config.py file need to be modified. The values are respectively the directory of coco data set, the directory name of data set used for reasoning, and the annotation file used for precision calculation after reasoning. instances_set is concatenated with val_data_type to ensure that the file is correct and exists.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

## Model Description

### Performance

#### Train Performance

| Parameter           | Ascend                                                       |
| ------------------- | ------------------------------------------------------------ |
| Model Version       | Retinanet                                                    |
| Resource            | Ascend 910；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8 |
| uploaded Date       | 02/10/2021(month/day/year)                                   |
| Luojianet Version   | 1.0.6                                                        |
| Dataset             | DOTA-V1.5                                                    |
| Batch_size          | 32                                                           |
| Training Parameters | src/config.py                                                |
| Optimizer           | Momentum                                                     |
| Loss  Function      | Focal loss                                                   |
| Loss                | 0.4-0.6                                                      |
| Total time          | 8p: 12h                                                      |

#### Evaluation Performance

| Parameter        | Ascend                                                       |
| ---------------- | ------------------------------------------------------------ |
| Model Version    | Retinanet                                                    |
| Resource         | Ascend 910；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8 |
| uploaded Date    | 02/10/2021(month/day/year)                                   |
| Luojianet Vesion | 1.0.6                                                        |
| Dataset          | DOTA-V1.5                                                    |
| Batch_size       | 32                                                           |
| Speed            | 1p：50FPS                                                    |
| Map              | 52.2%                                                        |
| Total time       | 6min                                                         |

## Description of Random Situation

In the `dataset.py` script, we set random seeds in the `create_dataset` function. We also set random seeds in the `train.py` script.

