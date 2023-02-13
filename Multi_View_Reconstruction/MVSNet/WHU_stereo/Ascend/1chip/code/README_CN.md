# 目录

- [目录](#目录)
- [MVSNet说明](#MVSNet说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
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

# [MVSNet说明](#目录)

MVSNet(Multi-view Stereo Network) 是一种经典的端到端深度学习多视密集匹配模型，
用于对具有多视角影像的场景的密集匹配，不借助于核线影像和视差值，而直接从原始多视影像出发，
以一张中心影像和几张辅助影像，以及与影像对应的相机位姿参数作为输入，得到中心影像对应的深度图。 
MVSNet 将多视成像几何显式地编码进深度学习网络中，从而在几何约束条件的支持下，端到端的实现多
视影像间的密集匹配任务，是现有通用的多视密集匹配方法的基础和核心架构。

[论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.html)：
Yao Y, Luo Z, Li S, et al. Mvsnet: Depth inference for unstructured multi-view stereo[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 767-783.

# [模型架构](#目录)

![](figs/network.png)

# [数据集](#目录)

使用的数据集：[WHU—MVS](http://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html)  
支持的数据集：[WHU—MVS]或与WHU-MVS格式相同的数据集  


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

建议用户使用WHU-MVS数据集来体验模型，
其他数据集需要使用与WHU-MVS相同的格式。

# [环境要求](#目录)

- 硬件 Ascend
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [LuoJiaNet](https://www.mindspore.cn/install)
- 更多关于LuojiaNet的信息，请查看以下资源：
    - [LuoJiaNet教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [LuoJiaNet Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

- 通过官方网站安装LuoJiaNet后，您可以按照如下步骤进行训练和评估：

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

  ```text
  # 在Ascend上训练8卡
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
└─MVSNet
  ├─README.md
  ├─README_CN.md
  ├─figs
    ├─network.png
    ├─result.png 
  ├─scripts
    ├─eval.sh                         # 在Ascend中启动单机训练（1卡）
    ├─predict.sh                      # 在Ascend中启动分布式训练（8卡）
    ├─train.sh                        # 在Ascend中启动单机训练
  ├─src
    ├─dataset.py                      # 数据读取文件
    ├─homography_warping.py           # 特征提取模块
    ├─loss.py                         # loss函数
    ├─module.py                       # 网络中间层
    ├─mvsnet.py                       # mvsnet网络模型
    ├─preprocess.py                   # 数据预处理模块
  ├─eval.py                           # 评估测试结果
  ├─predict.py                        # 推理端代码
  └─train.py                          # 训练网络
```

## [脚本参数](#目录)

train.py中主要参数如下：

```text
可选参数：
  --data_root           训练数据集目录
  --logdir              ckpt文件输出位置
  --normalize           对中心影像的处理方式
  --view_num            每次输入影像数
  --ndepths             深度值个数
  --max_w               图像最大宽度
  --max_h               图像最大高度
  --resize_scale        深度和图像的重采样范围
  --sample_scale        下采样范围
  --interval_scale      间隔范围
  --batch_size          每次训练的batch个数
  --adaptive_scaling    使图像尺寸拟合网络
  --epochs              训练的epoch个数
  --lr                  学习率的大小
  --decay_rate          学习率降低的幅度
  --decay_step          学习率在每多少个step降低
  --seed                随机数生成的种子
```

## [训练过程](#目录)

### 训练

在Ascend设备上，使用命令行语句执行单机训练示例（1卡）

```
python train.py --data_root='/mnt/gj/stereo' --view_num=3 --ndepths=200 --max_w=768 --max_h=384 --epochs=50 --lr=0.001
```

上述python命令将在后台运行，您可以通过控制台查看结果。

训练结束后，您可在指定的输出文件夹下找到checkpoint文件。 得到如下损失值：

```text
# grep "loss:" train/log.txt
INFO:epoch[1], iter[1], loss:5.710720062255859
INFO:epoch[1], iter[100], loss:0.36795997619628906
INFO:epoch[1], iter[200], loss:0.10597513616085052
INFO:epoch[1], iter[300], loss:0.06676928699016571
INFO:epoch[1], iter[400], loss:0.03752899169921875
INFO:epoch[1], iter[500], loss:0.010647434741258621
...
```

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
=============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.635
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.717
```

### Test-dev

```bash
python test.py \
    --data_dir=./dataset/coco2017 \
    --pretrained=yolov4.ckpt \
    --testing_shape=608 > log.txt 2>&1 &
OR
bash run_test.sh dataset/coco2017 checkpoint/yolov4.ckpt
```

predict_xxx.json文件位于test/outputs/%Y-%m-%d_time_%H_%M_%S/。
将文件predict_xxx.json重命名为detections_test-dev2017_yolov4_results.json，并将其压缩为detections_test-dev2017_yolov4_results.zip。
将detections_test-dev2017_yolov4_results.zip文件提交到MS COCO评估服务器用于test-dev 2019 (bbox) <https://competitions.codalab.org/competitions/20794#participate>。
您将在文件末尾获得这样的结果。查看评分输出日志。

```text
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.642
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
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

在执行推理之前，必须在910环境上通过导出脚本导出MINDIR文件。
当前批处理大小只能设置为1。 精度计算过程需要70G+内存空间。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID] [ANN_FILE]
```

`DEVICE_ID`是可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在ac.log文件中找到类似如下结果。

```text
=============coco eval reulst=========
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.438
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.630
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.475
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.330
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.542
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.588
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.410
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.716
```

# [模型说明](#目录)

## [性能](#目录)

### 评估性能

YOLOv4应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

|参数| YOLOv4 |
| -------------------------- | ----------------------------------------------------------- |
|资源| Ascend 910；CPU 2.60GHz, 192核；内存：755G；系统：EulerOS 2.8；|
|上传日期|2020年10月16日|
| MindSpore版本|1.0.0-alpha|
|数据集|118000张图像|
|训练参数|epoch=320, batch_size=8, lr=0.012,momentum=0.9|
| 优化器                  | Momentum                                                    |
|损失函数|Sigmoid Cross Entropy with logits, Giou Loss|
|输出|框和标签|
|损失| 50 |
|速度| 1卡：53FPS；8卡：390FPS (shape=416) 220FPS (动态形状)|
|总时长|48小时（动态形状）|
|微调检查点|约500M（.ckpt文件）|
|脚本| <https://gitee.com/mindspore/models/tree/master/official/cv/yolov4> |

### 推理性能

YOLOv4应用于20000张图像上（标注和数据格式必须与COCO test 2017相同）

|参数| YOLOv4 |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存：755G             |
|上传日期|2020年10月16日|
| MindSpore版本|1.0.0-alpha|
|数据集|20000张图像|
|批处理大小|1|
|输出|边框位置和分数，以及概率|
|精度|map >= 44.7%(shape=608)|
|推理模型|约500M（.ckpt文件）|

# [随机情况说明](#目录)

在dataset.py中，我们设置了“create_dataset”函数内的种子。
在var_init.py中，我们设置了权重初始化的种子。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
