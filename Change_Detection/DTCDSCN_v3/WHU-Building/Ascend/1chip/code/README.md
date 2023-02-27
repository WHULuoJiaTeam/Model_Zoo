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

This is a dual-task constrained deep Siamese convolutional network (DTCDSCN) model, which contains three subnetworks: a change detection network and two semantic segmentation networks.  DTCDSCN can accomplish both change detection and semantic segmentation at the same time, which can help to learn more discriminative object-level features and obtain a complete change detection map.  

[Paper:](https://doi.org/10.1109/LGRS.2020.2988032) 
Y. Liu, C. Pang, Z. Zhan, X. Zhang and X. Yang, "Building Change Detection for Remote Sensing Images Using a Dual-Task Constrained Deep Siamese Convolutional Network Model," in IEEE Geoscience and Remote Sensing Letters, vol. 18, no. 5, pp. 811-815, May 2021, doi: 10.1109/LGRS.2020.2988032.

Github:
[https://github.com/fitzpchao/DTCDSCN](https://github.com/fitzpchao/DTCDSCN)   

## [Model Architecture](#contents)

![Network Figure](image.png)

## [Dataset](#contents)

1. Download remote sensing building change detection image dataset, such as WHU ChangeDetection, LEVIR-CD, etc.
  The dataset used for this example：[WHU Building change detection dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 

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
  ├─module.py                       # DTCDSCN network model
  ├─cdloss.py                       # Loss function
  ├─eval.py                         # Evaluating results
  ├─prediction.py                   # Inference side code
  ├─config.py                       # Model configuration
  └─train.py                        # Training the network
```


### [Training Process](#contents)

#### [Script Parameters](#contents)

Major parameters ``config.py`` as follows:

```
    "device_target":"Ascend",     #GPU \ CPU \ Ascend
    "device_id": 0, # device ID
    "dataset_path": "/cache/data/",  # datset path
    "save_checkpoint_path": "/cache/checkpoint",  # save checkpoint path
    "resume":False,  # Whether to load pretrained model to train
    "batch_size": 4,
    "aug" : True,
    "step_per_epoch": 200,
    "epoch_size": 200, 
    "save_checkpoint": True, # Whether to save checkpoint
    "save_checkpoint_epochs": 200, # Save the model for every xx epoches
    "keep_checkpoint_max": 10, # The maximum number of models to save
    "decay_epochs": 20, # The number of epochs that the learning rate decays
    "max_lr": 0.001, # Maximum learning rate
    "min_lr": 0.00001 # Minimum learning rate
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

Run``python prediction.py --checkpoint_path **** --dataset_path ****`` or ``python prediction.py --checkpoint_path **** --left_input_file **** --right_input_file ****`` on the terminal to inference, with the following parameters:

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
