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

The residual neural network (ResNet) was proposed by He Kaiming, Zhang Xiangyu, Ren Shaoqing, Sun Jian, etc. of Microsoft Research Institute. The main contribution of residual neural network is the discovery of "degradation", and the invention of "shortcut connection" for the degradation phenomenon, which greatly eliminates the problem of training difficulty of neural network with excessive depth. The "depth" of neural network has broken through 100 layers for the first time, and the largest neural network has even exceeded 1000 layers.

[Paper Link](https://arxiv.org/abs/1512.03385):
He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015): 770-778.

## [Model Architecture](#contents)

![Network Figure](image.png)

## [Dataset](#contents)

Dataset used：[RSSCN7](https://sites.google.com/site/qinzoucn/documents)  
Annotation support：[RSSCN7]or annotation as the same format as RSSCN7

- The directory structure is as follows：
    ```text
        ├── dataset
            ├── RSSCN7
               ├── class1
               │    ├── 000000000001.jpg
               │    ├── 000000000002.jpg
               │    ├── ...
               ├── class2
               │    ├── 000000000001.jpg
               │    ├── 000000000002.jpg
               │    ├── ...
               ├── class3
               │    ├── 000000000001.jpg
               │    ├── 000000000002.jpg
               │    ├── ...
               ├── classN
               ├── ...
    ```

## [Environment Requirements](#contents)

This code is Huawei Modelarts Ascend platform **8P** version

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
  # Train 8p with Ascend
  # (1) Upload or copy your pretrained model to S3 bucket.
  # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (3) Set the code directory to "/path/ResNet" on the website UI interface.
  # (4) Set the startup file to "train.py" on the website UI interface.
  # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (6) Create your job.
  #
  # Eval 8p with Ascend
  # (1) Upload or copy your pretrained model to S3 bucket.
  # (2) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (3) Set the code directory to "/path/ResNet" on the website UI interface.
  # (4) Set the startup file to "eval.py" on the website UI interface.
  # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (6) Create your job.
  ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
└─ResNet
  ├─README.md
  ├─README_CN.md
  ├─Resnet.py                       # Resnet network model
  ├─config.py                       # Model configuration
  ├─utils.py                        # Data reading function, loss function, etc
  ├─test.py                         # Eval net
  ├─eval.py                         # Inference net
  └─train.py                        # Train net
```


### [Training Process](#contents)

#### [Script Parameters](#contents)

Major parameters ``config.py`` as follows:

```    
    "device_target":"Ascend",                          #GPU、CPU、Ascend
    "dataset_path": "RSSCN7/",                         # dataset path
    "save_checkpoint_path": "./checkpoint",            # save checkpoint path
    "resume":False,                                    # Whether to load pretrained model to train
    "class_num": 7,                                    # Types included in the data set
    "batch_size": 4,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "epoch_size": 350,                                  # training epoch
    "save_checkpoint": True,                            # Whether to save checkpoint
    "save_checkpoint_epochs": 1,                        # Save the model for every xx epoches
    "keep_checkpoint_max": 100,                         # The maximum number of models to save
    "opt": 'sgd',                                       #optimizer：rmsprop或sgd
    "opt_eps": 0.001, 
    "warmup_epochs": 50,                                # The number of epochs that the learning rate decays
    "lr_decay_mode": "warmup",                          #learning rate decays modes：steps、poly、cosine以及warmup
    "use_label_smooth": True, 
    "label_smooth_factor": 0.1,
    "lr_init": 0.0001,                                  # Init learning rate
    "lr_max": 0.001,                                    # Maximum learning rate
    "lr_end": 0.00001                                   # Minimum learning rate
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

Run ``python test.py --input_file **** --output_folder **** --checkpoint_path **** --classes_file ****``  on the terminal to inference, with the following parameters:

```
    --input_file, type=str, default=None, help='Input file path'
   --output_folder, type=str, default=None, help='Output file path'
   --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
   --classes_file, type=str, default=None, help='Classes saved txt path '
   --device_target, type=str, default="Ascend", help='Device target'
```

## [Description of Random Situation](#contents)

There are random seeds in ``eval.py`` and ``test.py`` files.

## [ModelZoo Homepage](#contents)

Please check the [Model Zoo](https://github.com/WHULuoJiaTeam/Model_Zoo).
