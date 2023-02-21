# 目录

- [目录](#目录)
- [YOLOv4说明](#yolov4说明)
- [模型架构](#模型架构)
- [预训练模型](#预训练模型)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [CKPT](#ckpt)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
  - [脚本和示例代码](#脚本和示例代码)
  - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
    - [训练](#训练)
    - [分布式训练](#分布式训练)
    - [迁移学习](#迁移学习)
  - [评估过程](#评估过程)
    - [验证](#验证)
    - [Test-dev](#test-dev)
  - [转换过程](#转换过程)
    - [转换](#转换)
  - [推理过程](#推理过程)
    - [用法](#用法)
    - [结果](#结果)
- [模型说明](#模型说明)
  - [性能](#性能)
    - [评估性能](#评估性能)
    - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [YOLOv4说明](#目录)

YOLOv4作为先进的检测器，它比所有可用的替代检测器更快（FPS）并且更准确（MS COCO AP50 ... 95和AP50）。
本文已经验证了大量的特征，并选择使用这些特征来提高分类和检测的精度。
这些特性可以作为未来研究和开发的最佳实践。

[论文](https://arxiv.org/pdf/2004.10934.pdf)：
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [模型架构](#目录)

选择CSPDarknet53主干、SPP附加模块、PANet路径聚合网络和YOLOv4（基于锚点）头作为YOLOv4架构。

# [预训练模型](#目录)

YOLOv4需要CSPDarknet53主干来提取图像特征进行检测。 您可以从[这里](https://download.luojianet.cn/model_zoo/r1.2/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428.ckpt)获取到在ImageNet2012上训练的预训练模型。

# [数据集](#目录)

使用的数据集：[DOTA-V1.5](https://captain-whu.github.io/DOTA/dataset.html)。数据集被切分为为600*600像素大小， overlap为20%。您可以从以下链接下载数据预处理（切分、coco格式转换等）代码及切分后的DOTA数据集。

支持的数据集：[COCO2017]或与MS COCO格式相同的数据集  
支持的标注：[COCO2017]或与MS COCO相同格式的标注

* 数据集预处理：https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/preprocess.zip
* 数据集：https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/DOTA.zip

- 目录结构如下，由用户定义目录和文件的名称：

    ```text
        ├── dataset
            ├── YOLOv4
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

建议用户使用MS COCO数据集来体验模型，
其他数据集需要使用与MS COCO相同的格式。

# [环境要求](#目录)

- 硬件 Ascend
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [LuojiaNet](https://www.luojianet.cn/install)
- 更多关于LuojiaNet的信息，请查看以下资源：
    - [LuojiaNet教程](https://www.luojianet.cn/tutorials/zh-CN/master/index.html)
    - [LuojiaNet Python API](https://www.luojianet.cn/docs/zh-CN/master/index.html)

# [CKPT](#目录)

此处提供在DOTA-V1.5上训练得到的ckpt文件，您可以将其用于微调、评估以及推理测试。下载链接为：

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/yolov4/DOTA/mult/test1_lr0.012_bs8/train/2023-01-10_time_16_40_38/ckpt_0/0-320_120960.ckpt

# [快速入门](#目录)

- 通过官方网站安装luojianet后，您可以按照如下步骤进行训练和评估：
- 在运行网络之前，准备CSPDarknet53.ckpt和hccl_8p.json文件。
    - 请参考[预训练模型]。

    - 生成hccl_8p.json，运行utils/hccl_tools/hccl_tools.py脚本。  
      以下参数“[0-8)”表示生成0~7卡的hccl_8p.json文件。

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

- 本地运行

  ```text
  # training_shape参数用于定义网络图像形状，默认为
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
  # 意思是使用11种形状作为输入形状，或者可以设置某种形状。

  # 使用python命令执行单尺度训练示例（1卡）
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

  # 使用shell脚本执行单尺度单机训练示例（1卡）
  bash run_standalone_train.sh dataset/xxx cspdarknet53_backbone.ckpt

  # 在Ascend设备上，使用shell脚本执行多尺度分布式训练示例（8卡）
  bash run_distribute_train.sh dataset/xxx cspdarknet53_backbone.ckpt rank_table_8p.json

  # 使用python命令评估
  python eval.py \
      --data_dir=./dataset/xxx \
      --pretrained=yolov4.ckpt \
      --testing_shape=608 > log.txt 2>&1 &

  # 使用shell脚本评估
  bash run_eval.sh dataset/xxx checkpoint/xxx.ckpt
  ```

- [ModelArts](https://support.huaweicloud.com/modelarts/)上训练

  ```python
  # 在Ascend上训练8卡
  # 此处给出两种方法，方法一在modelarts云平台中定义参数传入即可，方法二在config文件中进行配置。推荐方法一。
  # 方法一
  # (1) 在train.py中将get_args()函数的相关参数设置自己的数据路径及超参数等。
  # (2) 将代码及数据集上传到obs桶中。
  # (3) 在网页上将启动文件设置为train.py。
  # (4) 在网页上设置对应的parser参数。可参考教程链接：https://support.huaweicloud.com/modelarts_faq/modelarts_05_0265.html
  # 方法二
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_pretrain/'"。
  #          在base_config.yaml文件中设置“pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_pretrain/”。
  #          在网站UI界面上添加“pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制预训练的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“train.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上训练1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_pretrain/'"。
  #          在base_config.yaml文件中设置“pretrained_backbone='/cache/checkpoint_path/cspdarknet53_backbone.ckpt'”。
  #          在base_config.yaml文件中设置“is_distributed=0”。
  #          在base_config.yaml文件中设置“warmup_epochs=4”。
  #          在base_config.yaml文件中设置“training_shape=416”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_pretrain/”。
  #          在网站UI界面上添加“pretrained_backbone=/cache/checkpoint_path/cspdarknet53_backbone.ckpt”。
  #          在网站UI界面添加“is_distributed=0”。
  #          在网站UI界面添加“warmup_epochs=4”。
  #          在网站UI界面添加“training_shape=416”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制预训练的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“train.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上评估1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_trained_ckpt/'"。
  #          在base_config.yaml文件中设置“pretrained='/cache/checkpoint_path/model.ckpt'”。
  #          在base_config.yaml文件中设置“is_distributed=0”。
  #          在base_config.yaml文件中设置“"per_batch_size=1”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_trained_ckpt/”。
  #          在网站UI界面上添加“pretrained=/cache/checkpoint_path/model.ckpt”。
  #          在网站UI界面添加“is_distributed=0”。
  #          在网站UI界面添加“per_batch_size=1”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制训练好的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“eval.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上测试1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_dir='/cache/data/coco/'”。
  #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_trained_ckpt/'"。
  #          在base_config.yaml文件中设置“pretrained='/cache/checkpoint_path/model.ckpt'”。
  #          在base_config.yaml文件中设置“is_distributed=0”。
  #          在base_config.yaml文件中设置“"per_batch_size=1”。
  #          在base_config.yaml文件中设置“test_nms_thresh=0.45”。
  #          在base_config.yaml文件中设置“test_ignore_threshold=0.001”。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_dir=/cache/data/coco/”。
  #          在网站UI界面上添加“checkpoint_url=s3://dir_to_your_trained_ckpt/”。
  #          在网站UI界面上添加“pretrained=/cache/checkpoint_path/model.ckpt”。
  #          在网站UI界面添加“is_distributed=0”。
  #          在网站UI界面添加“per_batch_size=1”。
  #          在网站UI界面添加“test_nms_thresh=0.45”。
  #          在网站UI界面添加“test_ignore_threshold=0.001”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制训练好的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/yolov4”。
  # （6）在网站UI界面上设置启动文件为“test.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  ```

# [脚本说明](#目录)

## [脚本和示例代码](#目录)

```text
└─yolov4
  ├─README.md
  ├─README_CN.md
  ├─luojianet_hub_conf.py             # luojianet Hub配置
  ├─scripts
    ├─run_standalone_train.sh         # 在Ascend中启动单机训练（1卡）
    ├─run_distribute_train.sh         # 在Ascend中启动分布式训练（8卡）
    └─run_eval.sh                     # 在Ascend中启动评估
    ├─run_test.sh                     # 在Ascend中启动测试
  ├─src
    ├─__init__.py                     # Python初始化文件
    ├─config.py                       # 参数配置
    ├─cspdarknet53.py                 # 网络主干
    ├─distributed_sampler.py          # 数据集迭代器
    ├─export.py                       # 将luojianet模型转换为MINDIR,AIR模型
    ├─initializer.py                  # 参数初始化器
    ├─logger.py                       # 日志函数
    ├─loss.py                         # 损失函数
    ├─lr_scheduler.py                 # 生成学习率
    ├─transforms.py                   # 预处理数据
    ├─util.py                         # 工具函数
    ├─yolo.py                         # YOLOv4网络
    ├─yolo_dataset.py                 # 为YOLOv4创建数据集
  ├─eval.py                           # 评估验证结果
  ├─test.py#                          # 评估测试结果
  └─train.py                          # 训练网络
```

## [脚本参数](#目录)

train.py中主要参数如下：

```text
可选参数：
  -h, --help            显示此帮助消息并退出
  --device_target       实现代码的设备：“Ascend”为默认值
  --data_dir DATA_DIR   训练数据集目录
  --per_batch_size PER_BATCH_SIZE
                        训练的批处理大小。 默认值：8。
  --pretrained_backbone PRETRAINED_BACKBONE
                        CspDarkNet53的ckpt文件。 默认值：""。
  --resume_yolov4 RESUME_YOLOV4
                        YOLOv4的ckpt文件，用于微调。
                        默认值：""
  --lr_scheduler LR_SCHEDULER
                        学习率调度器，取值选项：exponential，
                        cosine_annealing。 默认值：exponential
  --lr LR               学习率。 默认值：0.001
  --lr_epochs LR_EPOCHS
                        LR变化轮次，用“,”分隔。
                        默认值：220,250
  --lr_gamma LR_GAMMA   将LR降低一个exponential lr_scheduler因子。
                        默认值：0.1
  --eta_min ETA_MIN     cosine_annealing调度器中的eta_min。 默认值：0
  --T_max T_MAX         cosine_annealing调度器中的T-max。 默认值：320
  --max_epoch MAX_EPOCH
                        训练模型的最大轮次数。 默认值：320
  --warmup_epochs WARMUP_EPOCHS
                        热身轮次。 默认值：0
  --weight_decay WEIGHT_DECAY
                        权重衰减因子。 默认值：0.0005
  --momentum MOMENTUM   动量。 默认值：0.9
  --loss_scale LOSS_SCALE
                        静态损失尺度。 默认值：64
  --label_smooth LABEL_SMOOTH
                        CE中是否使用标签平滑。 默认值：0
  --label_smooth_factor LABEL_SMOOTH_FACTOR
                        原one-hot的光滑强度。 默认值：0.1
  --log_interval LOG_INTERVAL
                        日志记录间隔步数。 默认值：100
  --ckpt_path CKPT_PATH
                        Checkpoint保存位置。 默认值：outputs/
  --ckpt_interval CKPT_INTERVAL
                        保存checkpoint间隔。 默认值：None
  --is_save_on_master IS_SAVE_ON_MASTER
                        在master或all rank上保存ckpt，1代表master，0代表
                        all ranks。 默认值：1
  --is_distributed IS_DISTRIBUTED
                        是否分发训练，1代表是，0代表否。 默认值：
                        1
  --rank RANK           分布式本地进程序号。 默认值：0
  --group_size GROUP_SIZE
                        设备进程总数。 默认值：1
  --need_profiler NEED_PROFILER
                        是否使用profiler。 0表示否，1表示是。 默认值：0
  --training_shape TRAINING_SHAPE
                        恢复训练形状。 默认值：""
  --resize_rate RESIZE_RATE
                        多尺度训练的缩放速率。 默认值：10
  --transfer_train TRANSFER_TRAIN
                        是否在其他数据集上进行迁移学习, 如果设置True filter_weight功能也开启。 默认值：False
```

## [训练过程](#目录)

可以从头开始训练YLOv4，也可以使用cspdarknet53主干训练。
Cspdarknet53是一个分类器，可以在ImageNet(ILSVRC2012)等数据集上训练。
用户可轻松训练Cspdarknet53。 只需将分类器Resnet50的主干替换为cspdarknet53。
可在luojianet ModelZoo中轻松获取Resnet50。

### 训练

在Ascend设备上，使用shell脚本执行单机训练示例（1卡）

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

上述python命令将在后台运行，您可以通过log.txt文件查看结果。

训练结束后，您可在默认输出文件夹下找到checkpoint文件。 得到如下损失值：

```text

# grep "loss:" train/log.txt
2023-01-10 17:29:07,004:INFO:epoch[1], iter[377], loss:1280.677487, per step time: 7642.25 ms, fps: 8.37, lr:0.003000000026077032
2023-01-10 17:32:11,619:INFO:epoch[2], iter[755], loss:681.348236, per step time: 488.40 ms, fps: 131.04, lr:0.006000000052154064
2023-01-10 17:35:15,484:INFO:epoch[3], iter[1133], loss:595.682223, per step time: 486.41 ms, fps: 131.58, lr:0.008999999612569809
2023-01-10 17:38:18,124:INFO:epoch[4], iter[1511], loss:566.903601, per step time: 483.17 ms, fps: 132.46, lr:0.012000000104308128
...
```

### 分布式训练

在Ascend设备上，使用shell脚本执行分布式训练示例（8卡）

```bash
bash run_distribute_train.sh dataset/coco2017 cspdarknet53_backbone.ckpt rank_table_8p.json
```

上述shell脚本将在后台运行分布式训练。 您可以通过train_parallel[X]/log.txt文件查看结果。 得到如下损失值：

```text
# 分布式训练结果(8卡，动态形状)
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

### 迁移学习

可以基于预训练分类或检测模型来训练自己的模型。 按照以下步骤进行迁移学习。

1. 将数据集转换为COCO样式。 否则，必须添加自己的数据预处理代码。
2. 修改 `default_config.yaml` 文件:
   1) 根据适配的数据集修改`labels`。
   2) 修改`transfer_train` 为 `True` 开启迁移学习功能。
   3) `pretrained_checkpoint` 用于指定加载的预训练权重，如果没有设置将会自动下载在coco数据集上预训练的权重。
   4) 修改`run_eval` 为 `True` 开启训练中验证集评估的功能。
3. 使用新的配置和参数构建自己的bash脚本。

## [评估过程](#目录)

### 验证

```bash
python eval.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_eval.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

上述python命令将在后台运行。 您可以通过log.txt文件查看结果。 测试数据集的mAP如下：

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

## [转换过程](#目录)

### 转换

如果您想推断Ascend 310上的网络，则应将模型转换为MINDIR：

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

必须设置ckpt_file参数。
`FILE_FORMAT`取值为["AIR", "ONNX", "MINDIR"]。

## [推理过程](#目录)

### 用法

方法一、使用inference.py进行推理

脚本支持GPU下的多个bacth_size推理，推荐设置batch_size=1，避免漏检出现运行逻辑错误。使用方式如下：

```python
# GPU inference
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

方法二、使用.sh进行推理

在执行推理之前，必须在910环境上通过导出脚本导出MINDIR文件。
当前批处理大小只能设置为1。 精度计算过程需要70G+内存空间。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [ANN_FILE]
```

`DEVICE_ID`是可选参数，默认值为0。

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数          | YOLOv4                                                       |
| ------------- | ------------------------------------------------------------ |
| 资源          | Ascend 910；CPU 2.60GHz, 192核；内存：755G；系统：EulerOS 2.8； |
| 上传日期      | 2023年1月16日                                                |
| luojianet版本 | 1.0.6                                                        |
| 数据集        | DOTA-V1.5                                                    |
| 训练参数      | epoch=320, batch_size=8, lr=0.012,momentum=0.9               |
| 优化器        | Momentum                                                     |
| 损失函数      | Sigmoid Cross Entropy with logits, Giou Loss                 |
| 输出          | 框和标签                                                     |
| 损失          | 200                                                          |
| 速度          | 8卡： 500ms/step                                             |
| 总时长        | 17h22min                                                     |
| 微调检查点    | 约500M（.ckpt文件）                                          |

### 评估性能

|参数| YOLOv4 |
| -------------------------- | ----------------------------------------------------------- |
|资源| Ascend 910；CPU 2.60GHz, 192核；内存：755G；系统：EulerOS 2.8；|
|上传日期|2023年1月16日|
| luojianet版本|1.0.6|
|数据集|DOTA-V1.5|
|训练参数|epoch=320, batch_size=8, lr=0.012,momentum=0.9|
| 优化器                  | Momentum                                                    |
|损失函数|Sigmoid Cross Entropy with logits, Giou Loss|
|输出|框和标签|
|损失| 200                                                          |
|速度| 1卡：114FPS |
|总时长|29min|
|微调检查点|约500M（.ckpt文件）|
|Map|68.2%|

# [随机情况说明](#目录)

在dataset.py中，我们设置了“create_dataset”函数内的种子。
在var_init.py中，我们设置了权重初始化的种子。

