# Contents

- [YOLOv4 Description](#YOLOv4-description)
- [Model Architecture](#model-architecture)
- [Pretrain Model](#pretrain-model)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [CKPT](#ckpt)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv4 Description](#contents)

YOLOv4 is a state-of-the-art detector which is faster (FPS) and more accurate (MS COCO AP50...95 and AP50) than all available alternative detectors.
YOLOv4 has verified a large number of features, and selected for use such of them for improving the accuracy of both the classifier and the detector.
These features can be used as best-practice for future studies and developments.

[Paper](https://arxiv.org/pdf/2004.10934.pdf):
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [Model Architecture](#contents)

YOLOv4 choose CSPDarknet53 backbone, SPP additional module, PANet path-aggregation neck, and YOLOv4 (anchor based) head as the architecture of YOLOv4.

# [Pretrain Model](#contents)

YOLOv4 needs a CSPDarknet53 backbone to extract image features for detection. The pretrained checkpoint trained with ImageNet2012 can be downloaded at [hear](https://download.luojianet.cn/model_zoo/r1.2/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428.ckpt).

# [Dataset](#contents)

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

# [CKPT](#contents)

Here we give the .ckpt file. You can use this file to pretrain, eval and infer. The link is as follows:

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/yolov4/DOTA/mult/test1_lr0.012_bs8/train/2023-01-10_time_16_40_38/ckpt_0/0-320_120960.ckpt

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
- Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [Luojianet]([首页](http://58.48.42.237/luojiaNet/home))
- For more information, please check the resources below：
    - [Luojianet Tutorials]([初学入门](http://58.48.42.237/luojiaNet/tutorial/quickstart))
    - [Luojianet API]([API](http://58.48.42.237/luojiaNet/luojiaNetapi/))

# [Quick Start](#contents)

- After installing Luojianet via the official website, you can start training and evaluation as follows:
- Prepare the CSPDarknet53.ckpt and hccl_8p.json files, before run network.
    - Please refer to [Pretrain Model]

    - Genatating hccl_8p.json, Run the script of utils/hccl_tools/hccl_tools.py.
      The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

- Run on local

  ```text
  # The parameter of training_shape define image shape for network, default is
                     [416, 416],
                     [448, 448],
                     [480, 480],
                     [512, 512],
                     [544, 544],
                     [576, 576],
                     [608, 608],
                     [640, 640],
                     [672, 672],
                     [704, 704],
                     [736, 736].
  # It means use 11 kinds of shape as input shape, or it can be set some kind of shape.

  #run training example(1p) by python command (Training with a single scale)
  python train.py \
      --data_dir=./dataset/xxx \
      --pretrained_backbone=cspdarknet53_backbone.ckpt \
      --is_distributed=0 \
      --lr=0.1 \
      --t_max=320 \
      --max_epoch=320 \
      --warmup_epochs=4 \
      --training_shape=416 \
      --lr_scheduler=cosine_annealing > log.txt 2>&1 &

  # standalone training example(1p) by shell script (Training with a single scale)
  bash run_standalone_train.sh dataset/xxx cspdarknet53_backbone.ckpt

  # For Ascend device, distributed training example(8p) by shell script (Training with multi scale)
  bash run_distribute_train.sh dataset/xxx cspdarknet53_backbone.ckpt rank_table_8p.json

  # run evaluation by python command
  python eval.py \
      --data_dir=./dataset/xxx \
      --pretrained=yolov4.ckpt \
      --testing_shape=608 > log.txt 2>&1 &

  # run evaluation by shell script
  bash run_eval.sh dataset/xxx checkpoint/xxx.ckpt
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
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on base_config.yaml file.
  #          Set "pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_pretrain/" on the website UI interface.
  #          Add "pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your pretrained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "train.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Train 1p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_pretrain/'" on base_config.yaml file.
  #          Set "pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'" on base_config.yaml file.
  #          Set "is_distributed=0" on base_config.yaml file.
  #          Set "warmup_epochs=4" on base_config.yaml file.
  #          Set "training_shape=416" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_pretrain/" on the website UI interface.
  #          Add "pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt" on the website UI interface.
  #          Add "is_distributed=0" on the website UI interface.
  #          Add "warmup_epochs=4" on the website UI interface.
  #          Add "training_shape=416" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your pretrained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "train.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Eval 1p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on base_config.yaml file.
  #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
  #          Set "is_distributed=0" on base_config.yaml file.
  #          Set "per_batch_size=1" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
  #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
  #          Add "is_distributed=0" on the website UI interface.
  #          Add "per_batch_size=1" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your trained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "eval.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  #
  # Test 1p with Ascend
  # (1) Perform a or b.
  #       a. Set "enable_modelarts=True" on base_config.yaml file.
  #          Set "data_dir='/cache/data/coco/'" on base_config.yaml file.
  #          Set "checkpoint_url='s3://dir_to_your_trained_ckpt/'" on base_config.yaml file.
  #          Set "pretrained='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
  #          Set "is_distributed=0" on base_config.yaml file.
  #          Set "per_batch_size=1" on base_config.yaml file.
  #          Set "test_nms_thresh=0.45" on base_config.yaml file.
  #          Set "test_ignore_threshold=0.001" on base_config.yaml file.
  #          Set other parameters on base_config.yaml file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/coco/" on the website UI interface.
  #          Add "checkpoint_url=s3://dir_to_your_trained_ckpt/" on the website UI interface.
  #          Add "pretrained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
  #          Add "is_distributed=0" on the website UI interface.
  #          Add "per_batch_size=1" on the website UI interface.
  #          Add "test_nms_thresh=0.45" on the website UI interface.
  #          Add "test_ignore_threshold=0.001" on the website UI interface.
  #          Add other parameters on the website UI interface.
  # (3) Upload or copy your trained model to S3 bucket.
  # (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
  # (5) Set the code directory to "/path/yolov4" on the website UI interface.
  # (6) Set the startup file to "test.py" on the website UI interface.
  # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
  # (8) Create your job.
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─yolov4
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training(1p) in ascend
    ├─run_distribute_train.sh         # launch distributed training(8p) in ascend
    └─run_eval.sh                     # launch evaluating in ascend
    ├─run_test.sh                     # launch testing in ascend
  ├─src
    ├─__init__.py                     # python init file
    ├─config.py                       # parameter configuration
    ├─cspdarknet53.py                 # backbone of network
    ├─distributed_sampler.py          # iterator of dataset
    ├─export.py                       # convert luojianet model to air model
    ├─initializer.py                  # initializer of parameters
    ├─logger.py                       # log function
    ├─loss.py                         # loss function
    ├─lr_scheduler.py                 # generate learning rate
    ├─transforms.py                   # Preprocess data
    ├─util.py                         # util function
    ├─yolo.py                         # yolov4 network
    ├─yolo_dataset.py                 # create dataset for YOLOV4
  ├─eval.py                           # evaluate val results
  ├─test.py#                          # evaluate test results
  └─train.py                          # train net
```

## [Script Parameters](#contents)

Major parameters train.py as follows:

```text
optional arguments:
  -h, --help            show this help message and exit
  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --data_dir DATA_DIR   Train dataset directory.
  --per_batch_size PER_BATCH_SIZE
                        Batch size for Training. Default: 8.
  --pretrained_backbone PRETRAINED_BACKBONE
                        The ckpt file of CspDarkNet53. Default: "".
  --resume_yolov4 RESUME_YOLOV4
                        The ckpt file of YOLOv4, which used to fine tune.
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
  --t_max T_MAX         T-max in cosine_annealing scheduler. Default: 320
  --max_epoch MAX_EPOCH
                        Max epoch num to train the model. Default: 320
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Default: 0
  --weight_decay WEIGHT_DECAY
                        Weight decay factor. Default: 0.0005
  --momentum MOMENTUM   Momentum. Default: 0.9
  --loss_scale LOSS_SCALE
                        Static loss scale. Default: 64
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
                        Resize rate for multi-scale training. Default: 10
  --transfer_train TRANSFER_TRAIN
                        Transfer training on other dataset, if set it True set filter_weight True. Default: False
```

## [Training Process](#contents)

YOLOv4 can be trained from the scratch or with the backbone named cspdarknet53.
Cspdarknet53 is a classifier which can be trained on some dataset like ImageNet(ILSVRC2012).
It is easy for users to train Cspdarknet53. Just replace the backbone of Classifier Resnet50 with cspdarknet53.
Resnet50 is easy to get in luojianet model zoo.

### Training

For Ascend device, standalone training example(1p) by shell script

```bash
bash run_standalone_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt
```

```text
python train.py \
    --data_dir=/dataset/xxx \
    --pretrained_backbone=cspdarknet53_backbone.ckpt \
    --is_distributed=0 \
    --lr=0.1 \
    --t_max=320 \
    --max_epoch=320 \
    --warmup_epochs=4 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default. The loss value will be achieved as follows:

```text

# grep "loss:" train/log.txt
2023-01-10 17:29:07,004:INFO:epoch[1], iter[377], loss:1280.677487, per step time: 7642.25 ms, fps: 8.37, lr:0.003000000026077032
2023-01-10 17:32:11,619:INFO:epoch[2], iter[755], loss:681.348236, per step time: 488.40 ms, fps: 131.04, lr:0.006000000052154064
2023-01-10 17:35:15,484:INFO:epoch[3], iter[1133], loss:595.682223, per step time: 486.41 ms, fps: 131.58, lr:0.008999999612569809
2023-01-10 17:38:18,124:INFO:epoch[4], iter[1511], loss:566.903601, per step time: 483.17 ms, fps: 132.46, lr:0.012000000104308128
...
```

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```bash
bash run_distribute_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt rank_table_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt. The loss value will be achieved as follows:

```text
# distribute training result(8p, dynamic shape)
...
2023-01-10 17:29:07, 004: INF0:epoch[1], iter[377]， loss:1280. 677487，per step time: 7642. 25 ms, fps: 8. 37,
1r:0. 0000000026077032
2023-01-10 17:32:11, 619: INF0:epoch[2]，iter[755]， loss:681. 348236, per step time: 488. 40 ms, fps: 131. 04,
1r:0. 0000000052154064
2023-01-10 17:35:15, 484: INF0:epoch[3]，iter[1133]， loss:595. 682223, per step time: 486. 41 ms, fps: 131. 58,
1r :0.008999999612569809
2023-01-10 17:38:18, 124: INFO:epoch[4]，iter[1511]， loss:566. 903601, per step time: 483. 17 ms, fps: 132. 46,
1r:0. 012000000104308128
...
```

### Transfer Training

You can train your own model based on either pretrained classification model or pretrained detection model. You can perform transfer training by following steps.

1. Convert your own dataset to COCO style. Otherwise you have to add your own data preprocess code.
2. Change `default_config.yaml`:
   1) Set argument `labels` according to your own dataset.
   2) Set argument `transfer_train` to `True`, start to transfer training on other dataset.
   3) `pretrained_checkpoint` is the path of pretrained checkpoint, it will download pretrained checkpoint automatically if not set it.
   4) Set argument `run_eval` to `True` if you want get eval dataset mAP when training.
3. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation Process](#contents)

### Valid

```bash
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```text
# log.txt
2023-01-16 23:26:49,218:INFO:
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.682
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.410
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.464
2023-01-16 23:26:49,225:INFO:testing cost time 0.48 h
```

## [Convert Process](#contents)

### Convert

If you want to infer the network on Ascend 310, you should convert the model to MINDIR:

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --keep_detect [Bool]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]
`keep_detect` keep the detect module or not, default: True

## [Inference Process](#contents)

### Usage

Method 1. Use inference.py

This .py file support GPU and batch_size can be set randomly. We suggest you set the batch_size as 1 to avoid error caused by leak detection. The usage is as follows:

```python
# GPU inference
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

Method 2. Use .sh

Before performing inference, the mindir file must be exported by export script on the 910 environment.
Current batch_Size can only be set to 1. The precision calculation process needs about 70G+ memory space.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [ANN_FILE]
```

`DEVICE_ID` is optional, default value is 0.

# [Model Description](#contents)

## [Performance](#contents)

### Train Performance

| Parameters                 | YOLOv4                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G; System, Euleros 2.8; |
| uploaded Date              | 01/16/2023 (month/day/year)                                  |
| Luojianet Version          | 1.0.6                                                        |
| Dataset                    | DOTA-V1.5                                                    |
| Training Parameters        | epoch=320, batch_size=8, lr=0.012,momentum=0.9               |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                 |
| outputs                    | boxes and label                                              |
| Loss                       | 200                                                          |
| Speed                      | 8p: 500ms/step                                               |
| Total time                 | 17h22min                                                     |
| Checkpoint for Fine tuning | about 500M (.ckpt file)                                      |

### Evaluation Performance

| Parameters                 | YOLOv4                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G; System, Euleros 2.8; |
| uploaded Date              | 01/16/2023 (month/day/year)                                  |
| Luojianet Version          | 1.0.6                                                        |
| Dataset                    | DOTA-V1.5                                                    |
| Training Parameters        | epoch=320, batch_size=8, lr=0.012,momentum=0.9               |
| Optimizer                  | Momentum                                                     |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                 |
| outputs                    | boxes and label                                              |
| Loss                       | 200                                                          |
| Speed                      | 1p: 114FPS                                                   |
| Total time                 | 29min                                                        |
| Checkpoint for Fine tuning | about 500M (.ckpt file)                                      |
| Map                        | 68.2%                                                        |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.
In var_init.py, we set seed for weight initialization
