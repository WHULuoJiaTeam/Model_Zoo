# Contents

- [Contents](#contents)
    - [YOLOv3-DarkNet53 Description](#yolov3-darknet53-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [CKPT](#ckpt)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
            - [Distributed Training](#distributed-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
        - [Export MindIR](#export-mindir)
        - [Inference Process](#inference-process)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)

## [YOLOv3-DarkNet53 Description](#contents)

You only look once (YOLO) is a state-of-the-art, real-time object detection system. YOLOv3 is extremely fast and accurate.

Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.
YOLOv3 use a totally different approach. It apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

YOLOv3 uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more. The full details are in the paper!

[Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement. Joseph Redmon, Ali Farhadi,
University of Washington

## [Model Architecture](#contents)

YOLOv3 use DarkNet53 for performing feature extraction, which is a hybrid approach between the network used in YOLOv2, Darknet-19, and that newfangled residual network stuff. DarkNet53 uses successive 3 × 3 and 1 × 1 convolutional layers and has some shortcut connections as well and is significantly larger. It has 53 convolutional layers.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [DOTA-V1.5]([DOTA (captain-whu.github.io)](https://captain-whu.github.io/DOTA/dataset.html)). The image is clipped to 600*600 pixel size and 20% overlap. You can get the process code and processed dataset form here.

* Process code: https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/preprocess.zip
* Dataset（clipped）: https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/DOTA.zip

DOTA-V1.5 contains 16 common categories and 402,089 instances. Before use Yolov3 to train, please modify the dataset as coco data format. The directory structure is as follows. You can see the script description to know more. 
```text
    ├── dataset
        ├── DOTA(coco_root)
            ├── annotations
            │   ├─ train.json
            │   └─ val.json
            ├─ train
            │   ├─picture1.jpg
            │   ├─ ...
            │   └─picturen.jpg
            └─ val
                ├─picture1.jpg
                ├─ ...
                └─picturen.jpg
```

- If the user uses his own data set, the data set format needs to be converted to coco data format,
  and the data in the JSON file should correspond to the image data one by one.
  After accessing user data, because the size and quantity of image data are different,
  lr, anchor_scale and training_shape may need to be adjusted appropriately.

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
- Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [Luojianet]([首页](http://58.48.42.237/luojiaNet/home))
- For more information, please check the resources below：
    - [Luojianet Tutorials]([初学入门](http://58.48.42.237/luojiaNet/tutorial/quickstart))
    - [Luojianet API]([API](http://58.48.42.237/luojiaNet/luojiaNetapi/))

## [CKPT](#contents)

Here we give the .ckpt file. You can use this file to pretrain, eval and infer. The link is as follows:

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/yolov3/DOTA/mult/test1_lr0.0012/train/2023-01-10_time_10_34_46/ckpt_0/yolov3_320_189.ckpt

## [Quick Start](#contents)

- After installing Luojianet via the official website, you can start training and evaluation in as follows. If running on GPU, please add `--device_target=GPU` in the python command or use the "_gpu" shell script ("xxx_gpu.sh").
- Prepare the backbone_darknet53.ckpt and hccl_8p.json files, before run network.
    - Pretrained_backbone can use convert_weight.py, convert darknet53.conv.74 to luojianet ckpt.

      ```
      python convert_weight.py --input_file ./darknet53.conv.74
      ```

      darknet53.conv.74 can get from [download](https://pjreddie.com/media/files/darknet53.conv.74) .
      you can use command in linux os.

      ```
      wget https://pjreddie.com/media/files/darknet53.conv.74
      ```

    - Genarating hccl_8p.json, Run the script of utils/hccl_tools/hccl_tools.py.(Only useful for Ascend device)
      The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.
      
        - The name of json file generated by this command is hccl_8p_01234567_{host_ip}.json. For convenience of expression, use hccl_8p.json represents the json file.
      
      ```
      python hccl_tools.py --device_num "[0,8)"
      ```
    
- Train on local

  ```network
  # The parameter of training_shape define image shape for network, default is "".
  # It means use 10 kinds of shape as input shape, or it can be set some kind of shape.
  # run training example(1p) by python command.
  python train.py \
      --data_dir=./dataset/coco2014 \
      --pretrained_backbone=backbone_darknet53.ckpt \
      --is_distributed=0 \
      --lr=0.001 \
      --loss_scale=1024 \
      --weight_decay=0.016 \
      --T_max=320 \
      --max_epoch=320 \
      --warmup_epochs=4 \
      --training_shape=416 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &

  # For Ascend device, standalone training example(1p) by shell script
  bash run_standalone_train.sh dataset/coco2014 backbone_darknet53.ckpt

  # For GPU device, standalone training example(1p) by shell script
  bash run_standalone_train_gpu.sh dataset/coco2014 backbone_darknet53.ckpt

  # For Ascend device, distributed training example(8p) by shell script
  bash run_distribute_train.sh dataset/coco2014 backbone_darknet53.ckpt hccl_8p.json

  # For GPU device, distributed training example(8p) by shell script
  bash run_distribute_train_gpu.sh dataset/coco2014 backbone_darknet53.ckpt

  # run evaluation by python command
    - For the standalone training mode, the ckpt file generated by training is stored in train/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0 directory.
    - For distributed training mode, the ckpt file generated by training is stored in train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0 directory.

  python eval.py \
      --data_dir=./dataset/coco2014 \
      --pretrained=train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt \
      --testing_shape=416 > log.txt 2>&1 &

  # run evaluation by shell script
  bash run_eval.sh dataset/coco2014/ train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt
  ```

- Train on [ModelArts](https://support.huaweicloud.com/modelarts/)

  ```python
  # Train 8p with Ascend
  # Here we give two methods. Method 1 use parser, which you can set these parameters on modelarts.
  # Method 1.
  # (1) Set the appropriate parameters in the get_args() of train.py. The function of these parameters are explained in config.yaml.
  # (2) Upload your code and dataset to obs bucket.
  # (3) Set the startup file to "train.py" on the website UI interface.
  # (4) Set the parser arguments on the website UI interface. Here is the tutotial: https://support.huaweicloud.com/modelarts_faq/modelarts_05_0265.html
  # Method 2.
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco2014/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on base_config.yaml file.
  #          Set "pretrained_backbone='/cache/checkpoint_path/0-148_92000.ckpt'" on base_config.yaml file.
  #          Set "weight_decay=0.016" on base_config.yaml file.
  #          Set "warmup_epochs=4" on base_config.yaml file.
  #          Set "lr_scheduler='cosine_annealing'" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco2014/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_pretrain/" on the website UI interface.
  #          Add "pretrained_backbone=/cache/checkpoint_path/0-148_92000.ckpt" on the website UI interface.
  #          Add "weight_decay=0.016" on the website UI interface.
  #          Add "warmup_epochs=4" on the website UI interface.
  #          Add "lr_scheduler=cosine_annealing" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your pretrained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov3_darknet53" on the website UI interface.
  # (6) Set the startup file to "train.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Eval with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco2014/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on base_config.yaml file.
  #          Set "pretrained='/cache/checkpoint_path/0-320_102400.ckpt'" on base_config.yaml file.
  #          Set "testing_shape=416" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco2014/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
  #          Add "pretrained=/cache/checkpoint_path/0-320_102400.ckpt" on the website UI interface.
  #          Add "testing_shape=416" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your trained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov3_darknet53" on the website UI interface.
  # (6) Set the startup file to "eval.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```contents
.
└─yolov3_darknet53
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    ├─run_infer_310.sh                # launch inference in ascend
    └─run_eval.sh                     # launch evaluating in ascend
    ├─run_standalone_train_gpu.sh     # launch standalone training(1p) in gpu
    ├─run_distribute_train_gpu.sh     # launch distributed training(8p) in gpu
    ├─run_eval_gpu.sh                 # launch evaluating in gpu
    └─run_infer_gpu.sh                # launch ONNX inference in gpu
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─darknet.py                      # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov3 network
    ├─yolo_dataset.py                 # create dataset for YOLOV3
  ├─eval.py                           # eval net
  ├─eval_onnx.py                      # inference net
  └─train.py                          # train net
```

### [Script Parameters](#contents)

```parameters
Major parameters in train.py as follow.

optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 32.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The ckpt file of DarkNet53. Default: "".
  --resume_yolov3 RESUME_YOLOV3
                        The ckpt file of YOLOv3, which used to fine tune.
                        Default: ""
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler, options: exponential,
                        cosine_annealing. Default: exponential
  --lr LR               Learning rate. Default: 0.001
  --lr_epochs LR_EPOCHS
                        Epoch of changing of lr changing, split with ",".
                        Default: 220,250
  --lr_gamma LR_GAMMA   Decrease lr by a factor of exponential lr_scheduler.
                        Default: 0.1
  --eta_min ETA_MIN     Eta_min in cosine_annealing scheduler. Default: 0
  --T_max T_MAX         T-max in cosine_annealing scheduler. Default: 320
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 320
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 1024
  --label_smooth LABEL_SMOOTH
                        Whether to use label smooth in CE. Default:0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        Smooth strength of original one-hot. Default: 0.1
  --log_interval LOG_INTERVAL
                        Logging interval steps. Default: 100
  --ckpt_path CKPT_PATH
                        Checkpoint save location. Default: outputs/
  --ckpt_interval CKPT_INTERVAL
                        Save checkpoint interval. Default: None
  --is_save_on_master IS_SAVE_ON_MASTER
                        Save ckpt on master or all rank, 1 for master, 0 for
                        all ranks. Default: 1
  --is_distributed IS_DISTRIBUTED
                        Distribute train or not, 1 for yes, 0 for no. Default:
                        1
  --rank RANK           Local rank of distributed. Default: 0
  --group_size GROUP_SIZE
                        World size of device. Default: 1
  --need_profiler NEED_PROFILER
                        Whether use profiler. 0 for no, 1 for yes. Default: 0
  --training_shape TRAINING_SHAPE
                        Fix training shape. Default: ""
  --resize_rate RESIZE_RATE
                        Resize rate for multi-scale training. Default: None
  --bind_cpu BIND_CPU
                        Whether bind cpu when distributed training. Default: True
  --device_num DEVICE_NUM
                        Device numbers per server. Default: 8
```

### [Training Process](#contents)

#### Training

```command
python train.py \
    --data_dir=./dataset/coco2014 \
    --pretrained_backbone=backbone_darknet53.ckpt \
    --is_distributed=0 \
    --lr=0.001 \
    --loss_scale=1024 \
    --weight_decay=0.016 \
    --T_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

The python command above will run in the background, you can view the results through the file `log.txt`. If running on GPU, please add `--device_target=GPU` in the python command.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

```log
# grep "loss:" train/log.txt
epoch[1], iter[1], loss:13689.215820, fps:0.77 imgs/sec, lr:1.5873015399847645e-06, per step time: 165381.98685646057ms
epoch[2], iter[1], loss:303.118072, fps:14.70 imgs/sec, lr:1.5873015399847645e-06, per step time: 8706.60572203379ms
epoch[3], iter[1], loss:114.398706, fps:97.77 imgs/sec, lr:1.5873015399847645e-06, per step time: 1309.2485950106668ms
...
```

The model checkpoint will be saved in outputs directory.

#### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```command
bash run_distribute_train.sh dataset/coco2014 backbone_darknet53.ckpt hccl_8p.json
```

For GPU device, distributed training example(8p) by shell script

```command
bash run_distribute_train_gpu.sh dataset/coco2014 backbone_darknet53.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train_parallel0/log.txt`. The loss value will be achieved as follows:

```log
# distribute training result(8p)
epoch[0], iter[0], loss:14623.384766, 1.23 imgs/sec, lr:7.812499825377017e-05
epoch[0], iter[100], loss:1486.253051, 15.01 imgs/sec, lr:0.007890624925494194
epoch[0], iter[200], loss:288.579535, 490.41 imgs/sec, lr:0.015703124925494194
epoch[0], iter[300], loss:153.136754, 531.99 imgs/sec, lr:0.023515624925494194
epoch[1], iter[400], loss:106.429322, 405.14 imgs/sec, lr:0.03132812678813934
...
epoch[318], iter[102000], loss:34.135306, 431.06 imgs/sec, lr:9.63797629083274e-06
epoch[319], iter[102100], loss:35.652469, 449.52 imgs/sec, lr:2.409552052995423e-06
epoch[319], iter[102200], loss:34.652273, 384.02 imgs/sec, lr:2.409552052995423e-06
epoch[319], iter[102300], loss:35.430038, 423.49 imgs/sec, lr:2.409552052995423e-06
...
```

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below. If running on GPU, please add `--device_target=GPU` in the python command or use the "_gpu" shell script ("xxx_gpu.sh").

```command
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2014/ train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

This the standard format from `pycocotools`, you can refer to [cocodataset](https://cocodataset.org/#detection-eval) for more detail. And the result is as follows: 

```eval log
# log.txt
2023-01-17 15:36:37,827:INFO:
=============coco eval result=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.209
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.336
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.357

2023-01-17 15:36:37,846:INFO:testing cost time 0.48 h
```

### [Export MindIR or ONNX](#contents)

Currently, batchsize can only set to 1.

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --keep_detect [Bool] --device_target=[DEVICE]
```

The ckpt_file parameter is required,
Currently,`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]
`keep_detect` keep the detect module or not, default: True
`device_target` should be in ["Ascend", "GPU", "CPU"], default: Ascend

### [Inference Process](#contents)

#### Usage

Method 1. Use inference.py

This .py file support GPU and batch_size can be set randomly. We suggest you set the batch_size as 1 to avoid error caused by leak detection. The usage is as follows:

```python
# GPU inference
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

Method 2. Use .sh

Before performing inference, the air or onnx file must be exported by export.py.
Current batch_Size can only be set to 1. Because the DVPP hardware is used for processing, the picture must comply with the JPEG encoding format, Otherwise, an error will be reported. For example, the COCO_val2014_000000320612.jpg in coco2014 dataset needs to be deleted.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0. DATA_PATH is evaluation data path, ANNO_PATH is annotation file path, json format. e.g., instances_val2014.json.

```shell
# onnx inference
bash run_infer_gpu.sh [DATA_PATH] [ONNX_PATH]
```

DATA_PATH is evaluation data path, include annotation file path, json format. e.g., instances_val2014.json.

### [Performance](#contents)

#### Train Performance

| Parameters                 | YOLO                                                        |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | YOLOv3                                                      |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 01/17/2023 (month/day/year)                                 |
| Luojianet Version          | 1.0.6                                                       |
| Dataset                    | DOTA-V1.5                                                   |
| Training Parameters        | epoch=320, batch_size=16, lr=0.0012, momentum=0.9           |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits                           |
| outputs                    | boxes and label                                             |
| Loss                       | 34                                                          |
| Speed                      | 1pc: 1200-1400 ms/step;                                     |
| Total time                 | 8pc: 22 hours                                               |
| Checkpoint for Fine tuning | 474M (.ckpt file)                                           |

#### Evaluation Performance

| Parameters                 | YOLO                                                        |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | YOLOv3                                                      |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 01/17/2023 (month/day/year)                                 |
| Luojianet Version          | 1.0.6                                                       |
| Dataset                    | DOTA-V1.5                                                   |
| Training Parameters        | epoch=320, batch_size=16, lr=0.0012, momentum=0.9           |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Sigmoid Cross Entropy with logits                           |
| outputs                    | boxes and label                                             |
| Loss                       | 34                                                          |
| Speed                      | 1pc: 177FPS                                                 |
| Total time                 | 8pc: 0.48 hours                                             |
| Checkpoint for Fine tuning | 474M (.ckpt file)                                           |
| Map                        | 50.6%                                                       |

## [Description of Random Situation](#contents)

There are random seeds in distributed_sampler.py, transforms.py, yolo_dataset.py files.
