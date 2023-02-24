# Readme

## Contents

- [Readme](#readme)
  - [Contents](#contents)
  - [Model Description](#model-description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Quick Start(Optional)](#quick-startoptional)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
      - [Script Parameters](#script-parameters)
      - [Training](#training)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
  - [Description of Random Situation](#description-of-random-situation)
  - [ModelZoo Homepage](#modelzoo-homepage)

## [Model Description](#contents)

This is a novel multi-task network based on the idea of transfer learning called Building_CD, which is less dependent on change detection samples by appropriately selecting high-dimensional features for sharing and a unique decoding module. Different from other multi-task change detection networks, with the help of a high-accuracy building mask, this network can fully utilize the prior information from building detection branches and further improve the change detection result through the proposed object-level refinement algorithm.

[Paper Link](https://doi.org/10.3390/rs14040957):
S. Gao, W. Li, K. Sun, J. Wei, Y. Chen, and X. Wang, “Built-Up Area Change Detection Using Multi-Task Network with Object-Level Refinement,” Remote Sensing, vol. 14, no. 4, p. 957, Feb. 2022, doi: 10.3390/rs14040957.

Code provider：[gaosong@whu.edu.cn](gaosong@whu.edu.cn)  

## [Model Architecture](#contents)

![Network Figure](image.png)

## [Dataset](#contents)

1. Download remote sensing building change detection image dataset, ***WHU Building change detection dataset*** is recommended
  The dataset used in this example：[WHU Building change detection dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 

2. Organize the data set into the following format:
**Note:** The picture names in **A, B, building_A, building_B, and label** must correspond!
Where, **A** and **building_A** are phase **A** image and corresponding building mask respectively, **B** and **building_B** are phase **B** image and corresponding building mask respectively, and **label** is the corresponding change mask

```
.. code-block::
        .
        └── image_folder_dataset_directory
             ├── A
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── B
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── building_A
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── building_B
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── label
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
```

## [Environment Requirements](#contents)

This code is Huawei Modelarts Ascend platform **1P** version

- Hardware
    - Prepare hardware environment with Ascend platform.
- Framework
    - [LuojiaNet](http://58.48.42.237/luojiaNet/)
- For more information, please check the resources below：
    - [LuojiaNet tutorials](http://58.48.42.237/luojiaNet/tutorial/quickstart/)
    - [LuojiaNet Python API](http://58.48.42.237/luojiaNet/luojiaNetapi/)

## [Quick Start(Optional)](#contents)

- After installing LuojiaNet via the official website, you can start training and evaluation as follows:

- Train on [ModelArts](https://support.huaweicloud.com/modelarts/)

 ```text
  # Train 1p with Ascend
  # (1) Upload or copy your pretrained model to S3 bucket.
  # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (3) Set the code directory to "/path/Building-CD" on the website UI interface.
  # (4) Set the startup file to "train.py" on the website UI interface.
  # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (6) Create your job.
  #
  # Eval 1p with Ascend
  # (1) Upload or copy your pretrained model to S3 bucket.
  # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (3) Set the code directory to "/path/Building-CD" on the website UI interface.
  # (4) Set the startup file to "prediction.py" on the website UI interface.
  # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (6) Create your job.
  ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
└─Building_CD
  ├─README.md
  ├─README_CN.md
  ├─dataset.py                      # Dataset IO
  ├─fpn.py                          # FPN network model
  ├─vgg.py                          # VGG network model
  ├─mainet.py                       # Building_CD network etmodel main part
  ├─finalnet.py                     # Building_CD network model final part
  ├─test.py                         # Test results
  ├─eval.py                         # Evaluate results
  ├─prediction.py                   # Inference side code
  ├─config.py                       # Model configuration
  └─train.py                        # Training the network
```


### [Training Process](#contents)

#### [Script Parameters](#contents)

Major parameters ``config.py`` as follows:

```
    "device_target":"Ascend",      #GPU、CPU、Ascend
    "device_id":0,   #device ID
    "dataset_path": "./CD_data",  # datset path
    "save_checkpoint_path": "./checkpoint",  # save checkpoint path
    "resume":False,   # Whether to load pretrained model to train
    "batch_size": 8,
    "aug" : True,
    "step_per_epoch": 200,
    "epoch_size": 200, 
    "save_checkpoint": True, # Whether to save checkpoint
    "save_checkpoint_epochs": 200, # Save the model for every xx epoches
    "keep_checkpoint_max": 5, # The maximum number of models to save
    "decay_epochs": 20, # The number of epochs that the learning rate decays
    "max_lr": 0.001, # Maximum learning rate
    "min_lr": 0.00001, # Minimum learning rate
    "LR":1e-4
```


#### [Training](#contents)

Run ``python train.py`` on the terminal for training


### [Evaluation Process](#contents)

Run ``python eval.py --checkpoint_path **** --dataset_path ****`` on the terminal to evaluate, with the following parameters:

```
    --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
    --dataset_path, type=str, default=None, help='Eval dataset path'
    --device_target, type=str, default=config.device_target, help='Device target'
    --device_id, type=int, default=config.device_id, help='Device id'
```

### [Inference Process](#contents)

Run ``python prediction.py --checkpoint_path **** --dataset_path ****`` or ``python prediction.py --checkpoint_path **** --left_input_file **** --right_input_file ****`` on the terminal to inference, with the following parameters:

```
    --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
    --dataset_path, type=str, default=None, help='Predict dataset path'
    --left_input_file, type=str, default=None, help='Pre-period image'
    --right_input_file, type=str, default=None, help='Post-period image'
    --output_folder, type=str, default="./result", help='Results path'
    --device_target, type=str, default=config.device_target, help='Device target'
    --device_id, type=int, default=config.device_id, help='Device id'
```

## [Description of Random Situation](#contents)

There are random seeds in ``eval.py`` and ``prediction.py`` files.

## [ModelZoo Homepage](#contents)

Please check the [Model Zoo](https://github.com/WHULuoJiaTeam/Model_Zoo).
