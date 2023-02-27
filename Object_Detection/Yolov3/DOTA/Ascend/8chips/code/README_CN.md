# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOv3-DarkNet53描述](#yolov3-darknet53描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [CKPT](#ckpt)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# YOLOv3-DarkNet53描述

You only look once（YOLO）是最先进的实时物体检测系统。YOLOv3非常快速和准确。

先前的检测系统重新利用分类器或定位器来执行检测，将模型应用于多个位置和尺度的图像。图像的高分区域被认为是检测。
 YOLOv3使用了完全不同的方法。该方法将单个神经网络应用于全图像，将图像划分为区域，并预测每个区域的边界框和概率。这些边界框由预测概率加权。

YOLOv3使用了一些技巧来改进训练，提高性能，包括多尺度预测、更好的主干分类器等等，详情见论文。

[论文](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement.Joseph Redmon, Ali Farhadi,
University of Washington

# 模型架构

YOLOv3使用DarkNet53执行特征提取，这是YOLOv2中的Darknet-19和残差网络的一种混合方法。DarkNet53使用连续的3×3和1×1卷积层，并且有一些快捷连接，而且DarkNet53明显更大，它有53层卷积层。

# 数据集

使用的数据集：[DOTA-V1.5](https://captain-whu.github.io/DOTA/dataset.html)。数据集被切分为为600*600像素大小， overlap为20%。您可以从以下链接下载数据预处理（切分、coco格式转换等）代码及切分后的DOTA数据集。

* 数据预处理：https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/preprocess.zip
* 数据集：https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/DOTA.zip

DOTA-V1.5包含16个常见类别和402,089个实例。在使用Yolov4进行训练之前，请将数据集修改为coco数据格式。目录结构如下所示。您可以查看脚本描述了解更多信息。

- 数据集的文件目录结构如下所示

    ```ext
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

- 如果，用户使用的是用户自己的数据集，则需要将数据集格式转化为coco数据格式，并且，json文件中的数据要和图片数据对应好。
  接入用户数据后，因为图片数据尺寸和数量不一样，lr、anchor_scale和training_shape可能需要适当调整。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [Luojianet]([首页](http://58.48.42.237/luojiaNet/home))
- 如需查看详情，请参见如下资源：
    - [Luojianet Tutorials]([初学入门](http://58.48.42.237/luojiaNet/tutorial/quickstart))
    - [Luojianet API]([API](http://58.48.42.237/luojiaNet/luojiaNetapi/))
    

# CKPT

此处提供在DOTA-V1.5上训练得到的ckpt文件，您可以将其用于微调、评估以及推理测试。下载链接为：

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/yolov3/DOTA/mult/test1_lr0.0012/train/2023-01-10_time_10_34_46/ckpt_0/yolov3_320_189.ckpt

# 快速入门

- 通过官方网站安装Luojianet后，您可以按照如下步骤进行训练和评估：如果在GPU上运行，请在python命令中添加`--device_target=GPU`，或者使用“_gpu”shell脚本（“xxx_gpu.sh”）。
- 在运行任务之前，需要准备backbone_darknet53.ckpt和hccl_8p.json文件。
    - 使用yolov3_darknet53路径下的convert_weight.py脚本将darknet53.conv.74转换成luojianet ckpt格式。

      ```command
      python convert_weight.py --input_file ./darknet53.conv.74
      ```

      可以从网站[下载](https://pjreddie.com/media/files/darknet53.conv.74) darknet53.conv.74文件。
      也可以在linux系统中使用指令下载该文件。

      ```command
      wget https://pjreddie.com/media/files/darknet53.conv.74
      ```

    - 可以运行models/utils/hccl_tools/路径下的hccl_tools.py脚本生成hccl_8p.json文件，下面指令中参数"[0, 8)"表示生成0-7的8卡hccl_8p.json文件。（仅适用于Ascend场景）
        - 该命令生成的json文件名为 hccl_8p_01234567_{host_ip}.json, 为了表述方便，统一使用hccl_8p.json表示该json文件。

      ```command
      python hccl_tools.py --device_num "[0,8)"
      ```

- 在本地进行训练

  ```constet
  # training_shape参数定义网络图像形状，默认为""。
  # 意思是使用10种形状作为输入形状，或者可以设置某种形状。
  # 通过python命令执行训练示例(1卡)。
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

  # 对于Ascend设备，shell脚本单机训练示例(1卡)
  bash run_standalone_train.sh dataset/coco2014 backbone_darknet53.ckpt

  # 对于GPU设备，shell脚本单机训练示例(1卡)
  bash run_standalone_train_gpu.sh dataset/coco2014 backbone_darknet53.ckpt

  # 对于Ascend设备，使用shell脚本分布式训练示例(8卡)
  bash run_distribute_train.sh dataset/coco2014 backbone_darknet53.ckpt hccl_8p.json

  # 对于GPU设备，使用shell脚本分布式训练示例(8卡)
  bash run_distribute_train_gpu.sh dataset/coco2014 backbone_darknet53.ckpt

  # 使用python命令评估
    - 对于standalone训练模式, 训练生成的ckpt文件存放在 train/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0 目录下。
    - 对于分布式训练模式, 训练生成的ckpt文件存放在 train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0 目录下。

  python eval.py \
      --data_dir=./dataset/coco2014 \
      --pretrained=train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt \
      --testing_shape=416 > log.txt 2>&1 &

  # 通过shell脚本运行评估
  bash run_eval.sh dataset/coco2014/ train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt
  ```

- 在 [ModelArts](https://support.huaweicloud.com/modelarts/) 上训练

  ```python
  # 在modelarts上进行8卡训练（Ascend）
  # 此处给出两种方法，方法一在modelarts云平台中定义参数传入即可，方法二在config文件中进行配置。推荐方法一。
  # 方法一
  # (1) 在train.py中将get_args()函数的相关参数设置自己的数据路径及超参数等。
  # (2) 将代码及数据集上传到obs桶中。
  # (3) 在网页上将启动文件设置为train.py。
  # (4) 在网页上设置对应的parser参数。可参考教程链接：https://support.huaweicloud.com/modelarts_faq/modelarts_05_0265.html
  # 方法二
  # (1) 执行a或者b
  #       a. 在 base_config.yaml 文件中配置 "enable_modelarts=True"
  #          在 base_config.yaml 文件中配置 "data_dir='/cache/data/coco2014/'"
  #          在 base_config.yaml 文件中配置 "checkpoint_url='s3://dir_to_your_pretrain/'"
  #          在 base_config.yaml 文件中配置 "pretrained_backbone='/cache/checkpoint_path/0-148_92000.ckpt'"
  #          在 base_config.yaml 文件中配置 "weight_decay=0.016"
  #          在 base_config.yaml 文件中配置 "warmup_epochs=4"
  #          在 base_config.yaml 文件中配置 "lr_scheduler='cosine_annealing'"
  #          在 base_config.yaml 文件中配置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 "data_dir=/cache/data/coco2014/"
  #          在网页上设置 "checkpoint_url=s3://dir_to_your_pretrain/"
  #          在网页上设置 "pretrained_backbone=/cache/checkpoint_path/0-148_92000.ckpt"
  #          在网页上设置 "weight_decay=0.016"
  #          在网页上设置 "warmup_epochs=4"
  #          在网页上设置 "lr_scheduler=cosine_annealing"
  #          在网页上设置 其他参数
  # (2) 上传你的预训练模型到 S3 桶上
  # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
  # (4) 在网页上设置你的代码路径为 "/path/deeplabv3"
  # (5) 在网页上设置启动文件为 "train.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
  #
  # 在modelarts上进行验证（Ascend）
  # (1) 执行a或者b
  #       a. 在 base_config.yaml 文件中配置 "enable_modelarts=True"
  #          在 base_config.yaml 文件中配置 "data_dir='/cache/data/coco2014/'"
  #          在 base_config.yaml 文件中配置 "checkpoint_url='s3://dir_to_your_trained_ckpt/'"
  #          在 base_config.yaml 文件中配置 "pretrained='/cache/checkpoint_path/0-320_102400.ckpt'"
  #          在 base_config.yaml 文件中配置 "testing_shape=416"
  #          在 base_config.yaml 文件中配置 其他参数
  #       b. 在网页上设置 "enable_modelarts=True"
  #          在网页上设置 "data_dir=/cache/data/coco2014/"
  #          在网页上设置 "checkpoint_url=s3://dir_to_your_trained_ckpt/"
  #          在网页上设置 "pretrained=/cache/checkpoint_path/0-320_102400.ckpt"
  #          在网页上设置 "testing_shape=416"
  #          在网页上设置 其他参数
  # (2) 上传你的预训练模型到 S3 桶上
  # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
  # (4) 在网页上设置你的代码路径为 "/path/deeplabv3"
  # (5) 在网页上设置启动文件为 "train.py"
  # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
  # (7) 创建训练作业
  ```

# 脚本说明

## 脚本及样例代码

```text
.
└─yolov3_darknet53
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend中启动单机训练(1卡)
    ├─run_distribute_train.sh         # 在Ascend中启动分布式训练(8卡)
    ├─run_infer_310.sh                # 在Ascend中启动推理
    └─run_eval.sh                     # 在Ascend中启动评估
    ├─run_standalone_train_gpu.sh     # 在GPU中启动单机训练(1卡)
    ├─run_distribute_train_gpu.sh     # 在GPU中启动分布式训练(8卡)
    ├─run_eval_gpu.sh                 # 在GPU中启动评估
    └─run_infer_gpu.sh                # 在GPU中启动ONNX推理
  ├─src
    ├─__init__.py                     # python初始化文件
    ├─config.py                       # 参数配置
    ├─darknet.py                      # 网络骨干
    ├─distributed_sampler.py          # 数据集迭代器
    ├─initializer.py                  #参数初始化器
    ├─logger.py                       # 日志函数
    ├─loss.py                         # 损失函数
    ├─lr_scheduler.py                 # 生成学习率
    ├─transforms.py                   # 预处理数据
    ├─util.py                         # 工具函数
    ├─yolo.py                         # yolov3网络
    ├─yolo_dataset.py                 # 为YOLOV3创建数据集
  ├─eval.py                           # 评估网络
  ├─eval_onnx.py                      # 推理网络
  └─train.py                          # 训练网络
```

## 脚本参数

```text
train.py中主要参数如下：

可选参数：
  -h, --help            显示此帮助消息并退出。
  --Device_target       实现代码的设备：“Ascend" | "GPU"。默认设置："Ascend"。
  --data_dir DATA_DIR   训练数据集目录。
  --per_batch_size PER_BATCH_SIZE
                        训练批次大小。默认设置：32。
  --pretrained_backbone PRETRAINED_BACKBONE
                        DarkNet53的ckpt文件。默认设置：""。
  --resume_yolov3 RESUME_YOLOV3
                        YOLOv3的ckpt文件，用于微调。默认设置：""。
  --lr_scheduler LR_SCHEDULER
                        学习率调度器，选项：exponential，cosine_annealing。默认设置：exponential。
  --lr LR               学习率。默认设置：0.001。
  --lr_epochs LR_EPOCHS
                        lr changing轮次，用“,”分隔。默认设置：220,250。
  --lr_gamma LR_GAMMA   降低lr的exponential lr_scheduler因子。默认设置：0.1。
  --eta_min ETA_MIN     cosine_annealing调度器中的eta_min。默认设置：0。
  --T_max T_MAX         cosine_annealing调度器中的t_max。默认设置：0。
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。默认设置：320。
  --warmup_epochs WARMUP_EPOCHS
                        热身轮次。默认设置：0。
  --weight_decay WEIGHT_DECAY
                        权重衰减因子。默认设置：0.0005。
  --momentum MOMENTUM   动量。默认设置：0.9。
  --loss_scale LOSS_SCALE
                        静态损失等级。默认设置：1024。
  --label_smooth LABEL_SMOOTH
                        CE中是否使用标签平滑。默认设置：0。
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        独热平滑强度。默认设置：0.1。
  --log_interval LOG_INTERVAL
                        日志记录迭代间隔。默认设置：100。
  --ckpt_path CKPT_PATH
                        检查点保存位置。默认设置：outputs/。
  --ckpt_interval CKPT_INTERVAL
                        保存检查点间隔。默认设置：None。
  --is_save_on_master IS_SAVE_ON_MASTER
                        在主进程序号或所有进程序号上保存ckpt。1为主进程序号， 0为所有进程序号。默认设置：1。
  --is_distributed IS_DISTRIBUTED
                        是否分布训练，1表示是，0表示否，默认设置：1。
  --rank RANK           分布式本地排名。默认设置：0。
  --group_size GROUP_SIZE
                        设备进程总数。默认设置：1。
  --need_profiler NEED_PROFILER
                        是否使用调优器。0表示否，1表示是。默认设置：0。
  --training_shape TRAINING_SHAPE
                        固定训练形状。默认设置：""。
  --resize_rate RESIZE_RATE
                        多尺度训练的调整率。默认设置：None。
  --bind_cpu BIND_CPU
                        多卡运行是否绑核。默认设置：True
  --device_num DEVICE_NUM
                        一台服务器有多少张卡。默认设置：8
```

## 训练过程

### 训练

```python
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

上述python命令将在后台运行，您可以通过`log.txt`文件查看结果。如果在GPU上运行，请在python命令中添加`--device_target=GPU`。

训练结束后，您可在默认输出文件夹下找到检查点文件。损失值的实现如下：

```text
# grep "loss:" train/log.txt
epoch[1], iter[1], loss:13689.215820, fps:0.77 imgs/sec, lr:1.5873015399847645e-06, per step time: 165381.98685646057ms
epoch[2], iter[1], loss:303.118072, fps:14.70 imgs/sec, lr:1.5873015399847645e-06, per step time: 8706.60572203379ms
epoch[3], iter[1], loss:114.398706, fps:97.77 imgs/sec, lr:1.5873015399847645e-06, per step time: 1309.2485950106668ms
...
```

模型检查点将会储存在输出目录。

### 分布式训练

对于Ascend设备，使用shell脚本分布式训练示例(8卡)

```shell script
bash run_distribute_train.sh dataset/coco2014 backbone_darknet53.ckpt hccl_8p.json
```

对于GPU设备，使用shell脚本分布式训练示例(8卡)

```shell script
bash run_distribute_train_gpu.sh dataset/coco2014 backbone_darknet53.ckpt
```

上述shell脚本将在后台运行分布训练。您可以通过`train_parallel0/log.txt`文件查看结果。损失值的实现如下：

```text
# 分布式训练示例(8卡)
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

## 评估过程

### 评估

运行以下命令。如果在GPU上运行，请在python命令中添加`--device_target=GPU`，或者使用“_gpu”shell脚本（“xxx_gpu.sh”）。

```python
python eval.py \
    --data_dir=./dataset/coco2014 \
    --pretrained=train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt \
    --testing_shape=416 > log.txt 2>&1 &
```

或者

```shell script
bash run_eval.sh dataset/coco2014/ train_parallel0/outputs/{year}-{month}-{day}_time_{hour}_{minute}_{second}/ckpt_0/0-99_31680.ckpt
```

上述python命令将在后台运行，您可以通过log.txt文件查看结果。测试数据集的mAP如下：

```text
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

```

## 导出mindir，onnx模型

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --keep_detect [Bool] --device_target=[DEVICE]
```

参数`ckpt_file` 是必需的，目前,`FILE_FORMAT` 必须在 ["AIR", "ONNX", "MINDIR"]中进行选择。
参数`keep_detect` 是否保留坐标检测模块, 默认为True。
参数`device_target` 默认为Ascend，目前,`DEVICE` 必须在 ["Ascend", "GPU", "CPU"]中进行选择。

## 推理过程

### 用法

方法一、使用inference.py进行推理

脚本支持GPU下的多个bacth_size推理，推荐设置batch_size=1，避免漏检出现运行逻辑错误。使用方式如下：

```python
# GPU inference
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

方法二、使用.sh进行推理

运行以下命令。如果在GPU上运行，请在python命令中添加`--device_target=GPU`。

在执行推理之前，需要通过export.py导出mindir或者onnx文件。
目前仅可处理batch_Size为1，由于使用了DVPP硬件进行图片处理，因此图片必须满足JPEG编码格式，否则将会报错。比如coco2014数据集中的COCO_val2014_000000320612.jpg需要删除。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。DATA_PATH为推理数据所在的路径，ANNO_PATH为数据注解文件，为json文件，如instances_val2014.json。

```shell
# onnx 推理
bash run_infer_gpu.sh [DATA_PATH] [ONNX_PATH]
```

DATA_PATH为推理数据所在的路径，路径下应包含数据注解文件，如instances_val2014.json。

# 模型描述

## 性能

### 训练性能

| 参数          | YOLO                                                     |
| ------------- | -------------------------------------------------------- |
| 模型版本      | YOLOv3                                                   |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2023-01-17                                               |
| Luojianet版本 | 1.0.6                                                    |
| 数据集        | DOTA-V1.5                                                |
| 训练参数      | epoch=320，batch_size=16，lr=0.0012，momentum=0.9        |
| 优化器        | Momentum                                                 |
| 损失函数      | 带logits的Sigmoid交叉熵                                  |
| 输出          | 边界框和标签                                             |
| 损失          | 34                                                       |
| 速度          | 8卡：1200-1400毫秒/步;                                   |
| 总时长        | 8卡：22小时                                              |
| 微调检查点    | 474M (.ckpt文件)                                         |

### 评估性能

| 参数                 | YOLO                                                        |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | YOLOv3                                                      |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             |
| 上传日期              | 2023-01-17                             |
| Luojianet版本 | 1.0.6                                               |
| 数据集                    | DOTA-V1.5                                           |
| 训练参数        | epoch=320，batch_size=32，lr=0.0012，momentum=0.9     |
| 优化器                  | Momentum                                                    |
| 损失函数              | 带logits的Sigmoid交叉熵                           |
| 输出                    | 边界框和标签                                             |
| 损失                       | 34                                                          |
| 速度                      | 1卡：177FPS                         |
| 总时长                 | 8卡：0.48小时                               |
| 微调检查点 | 474M (.ckpt文件)                                           |
| Map | 50.6% |

# 随机情况说明

在distributed_sampler.py、transforms.py、yolo_dataset.py文件中有随机种子。
