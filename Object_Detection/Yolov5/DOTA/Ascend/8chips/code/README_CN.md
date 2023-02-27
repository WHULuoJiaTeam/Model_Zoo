# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOv5描述](#yolov5描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [CKPT](#ckpt)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
  - [脚本及样例代码](#脚本及样例代码)
  - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
    - [训练](#训练)
    - [分布式训练](#分布式训练)
  - [评估过程](#评估过程)
    - [评估](#评估)
  - [推理过程](#推理过程)
    - [导入MindIR](#导入MindIR)
    - [GPU上推理](#GPU上推理)
    - [Ascend310上推理](#Ascend310上推理)
    - [导出ONNX](#导出ONNX)
    - [执行ONXX评估](#执行ONXX评估)
- [模型描述](#模型描述)
  - [性能](#性能)
    - [训练性能](#训练性能)
    - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)

<!-- /TOC -->

# [YOLOv5描述](#contents)

YOLOv5于2020年4月发布，在COCO数据集上实现了最先进的性能，用于对象检测。它是YoloV3的一个重要改进，在**Backbone**中实现了一个新的架构，在**Neck**中进行了修改，使**mAP**(平均平均精度)提高了**10%，**FPS**(每秒帧数)提高了**12%。

[code](https://github.com/ultralytics/yolov5)

# [模型架构](#contents)

YOLOv5网络主要由CSP和焦点作为主干,空间金字塔池(SPP)附加模块、PANet路径聚集颈和YOLOv3头。[CSP](https://arxiv.org/,1911.11929)是一种新的脊板,可以增强CNN的学习能力。[空间金字塔池化](https://arxiv.org/,1406.4729)块在CSP上添加,以增加接受场,并分离出最重要的上下文特征。在YOLOv3中使用的对象检测中,PANet不是特征金字塔网络(FPN),而是用于不同探测器级别的参数聚合的方法。为了更具有专用性,cspblenet53包含5个CSP模块,使用的是卷积层**C**， k = 3x3， s = 2x2 *,在PANet和SPP使用 **1x1,5x5,9x9,,13x13 池化层。

# [数据集](#contents)

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

# [CKPT](#contents)

此处提供在DOTA-V1.5上训练得到的ckpt文件，您可以将其用于微调、评估以及推理测试。下载链接为：

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/yolov5/DOTA/mult/test2_lr0.02_bs32/train/yolov5_320_94.ckpt

# [快速入门](#目录)

通过官方网站安装Luojianet后，您可以按照如下步骤进行训练和评估。

```python
#run training example(1p) on Ascend/GPU by python command
python train.py \
    --device_target="Ascend" \ # Ascend or GPU
    --data_dir=xxx/dataset \
    --is_distributed=0 \
    --yolov5_version='yolov5s' \
    --lr=0.01 \
    --max_epoch=320 \
    --warmup_epochs=4 > log.txt 2>&1 &
```

```python
# run 1p by shell script, please change `device_target` in config file to run on Ascend/GPU, and change `T_max`, `max_epoch`, `warmup_epochs` refer to contents of notes
bash run_standalone_train.sh [DATASET_PATH]

# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

```python
# run evaluation on Ascend/GPU by python command
python eval.py \
    --device_target="Ascend" \ # Ascend or GPU
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --pretrained="***/*.ckpt" \
    --eval_shape=640 > log.txt 2>&1 &
```

```python
# run evaluation by shell script, please change `device_target` in config file to run on Ascend/GPU
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

注意default_config.yaml是8p上yolov5的默认参数。“batchsize”和“lr”在Ascend和GPU上是不同的，请参见“scripts/run_distribute_train.sh”或“scripts/run_distribute_train_gpu.sh”中的设置。

# [脚本说明](#目录)

## 脚本及样例代码

```python
├── model_zoo
    ├── README.md                              
    ├── yolov5
        ├── README.md                         
        ├── scripts
        │   ├──docker_start.sh                 // 启动docker
        │   ├──run_distribute_train.sh         // 在Ascend中启动分布式训练（8卡）
        │   ├──run_distribute_train_gpu.sh     // 在GPU中启动分布式训练（8卡）
        │   ├──run_standalone_train.sh         // 在GPU中启动单卡训练
        │   ├──run_infer_310.sh                // 在Ascend中启动推理
        │   ├──run_eval.sh                     // 在Ascend中启动推理
        │   ├──run_eval_onnx.sh                // 在GPU中启动ONXX推理
        ├──model_utils
        │   ├──config.py                       // 参数配置
        │   ├──device_adapter.py               // 设备信息获取
        │   ├──local_adapter.py                // 设备信息获取
        │   ├──moxing_adapter.py               // Decorator
        ├── src
        │   ├──backbone.py                     // 网络骨干
        │   ├──distributed_sampler.py          // 数据集迭代器
        │   ├──initializer.py                  // 参数初始化器
        │   ├──logger.py                       // 日志函数
        │   ├──loss.py                         // 损失函数
        │   ├──lr_scheduler.py                 // 生成学习率
        │   ├──transforms.py                   // 预处理数据
        │   ├──util.py                         // 工具函数
        │   ├──yolo.py                         // yolov5网络
        │   ├──yolo_dataset.py                 // 为YOLOV5创建数据集
        ├── default_config.yaml                // 参数配置(yolov5s 8p)
        ├── train.py                           // 训练网络
        ├── eval.py                            //评估网络
        ├── eval_onnx.py                       // 推理网络
        ├── export.py                          // 导入网络
        ├── inference.py                       // 推理网络
```

## 脚本参数

```text
train.py中主要参数如下：

可选参数：

  --device_target       实现代码的设备：“Ascend" | "GPU"。默认设置："Ascend"。
  --data_dir            训练数据集目录。
  --per_batch_size      训练批次大小。默认设置：32(1p) ; 16(Ascend 8p);32(GPU 8p).
  --resume_yolov5       resume_yolov5
                        yolov5的ckpt文件。默认设置：""。
  --lr_scheduler        学习率调度器,选项：exponential，cosine_annealing。默认设置：exponential
  --lr                  学习率。默认设置：0.01(1p) 0.02(Ascend 8p) 0.025(GPU 8p)
  --lr_epochs            lr changing轮次，用“,”分隔。默认设置：220,250。
  --lr_gamma           降低lr的exponential lr_scheduler因子。默认设置：0.1。
  --eta_min            cosine_annealing调度器中的eta_min。默认设置：0。
  --t_max               cosine_annealing调度器中的t_max。默认设置：300(8p)。
  --max_epoch           训练模型的最大轮次数。默认设置：320。
  --warmup_epochs      热身轮次。默认设置：20(8p)
  --weight_decay        权重衰减因子。默认设置：0.0005。
  --momentum            动量。默认设置：0.9。
  --loss_scale          静态损失等级。默认设置：64
  --label_smooth        CE中是否使用标签平滑。默认设置：0。
  --label_smooth_factor 独热平滑强度。默认设置：0.1。
  --log_interval       日志记录迭代间隔。默认设置：100。
  --ckpt_path          检查点保存位置。默认设置：outputs/。
  --is_distributed     是否分布训练，1表示是，0表示否，默认设置：1。
  --rank               分布式本地排名。默认设置：0。
  --group_size         设备进程总数。默认设置：1。
  --need_profiler      是否使用调优器。0表示否，1表示是。默认设置：0。
  --training_shape      固定训练形状。默认设置：""。
  --resize_rate          多尺度训练的调整率。默认设置：None。
  --bind_cpu           多卡运行是否绑核。默认设置：True
  --device_num          一台服务器有多少张卡。默认设置：8
```

## 训练过程

### 训练

```python
#在Acend设备单卡上执行python脚本
python train.py \
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --is_distributed=0 \
    --lr=0.01 \
    --T_max=320
    --max_epoch=320 \
    --warmup_epochs=4 \
    --per_batch_size=32 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

在GPU上运行训练1p时，您需要微调参数。

上面的python命令将在后台运行，您可以通过' log.txt '文件查看结果。

训练之后，默认情况下，您将在**outputs**文件夹下获得一些检查点文件。损失值将实现如下:

```text
# grep "loss:" log.txt
2023-01-10 09:57:08,545:INFO:epoch[1], iter[1], loss:7776.806641, fps:0.95 imgs/sec, lr:1.0638297680998221e-05, per step time: 270274.2323875427ms
2023-01-10 09:59:21,231:INFO:epoch[2], iter[1], loss:723.615182, fps:181.37 imgs/sec, lr:1.0638297680998221e-05, per step time: 1411.4941028838462ms
2023-01-10 10:00:25,981:INFO:epoch[3], iter[1], loss:206.321586, fps:371.70 imgs/sec, lr:1.0638297680998221e-05, per step time: 688.732755945084ms
2023-01-10 10:01:37,555:INFO:epoch[4], iter[1], loss:186.602023, fps:336.22 imgs/sec, lr:1.0638297680998221e-05, per step time: 761.4078471001159ms
2023-01-10 10:02:49,252:INFO:epoch[5], iter[1], loss:179.163211, fps:335.72 imgs/sec, lr:1.0638297680998221e-05, per step time: 762.5398077863327ms
...
```

### 分布式训练

通过以下脚本采用8卡训练：

```bash
# 在Ascend上执行8卡训练
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# 在GPU上执行8卡训练
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

上面的shell脚本将在后台进行分布式训练。您可以通过文件train_并行[X]/ log.txt(Ascend)或distribute_train / nohuttout(GPU)来查看结果。损失日志如下:

```text
# distribute training result(8p, dynamic shape)
...
023-01-10 09:57:08,532:INFO:epoch[1], iter[1], loss:7808.681641, fps:0.96 imgs/sec, lr:1.0638297680998221e-05, per step time: 266538.4922027588ms
2023-01-10 09:59:21,229:INFO:epoch[2], iter[1], loss:718.493260, fps:181.36 imgs/sec, lr:1.0638297680998221e-05, per step time: 1411.558856355383ms
2023-01-10 10:00:25,974:INFO:epoch[3], iter[1], loss:211.901843, fps:371.70 imgs/sec, lr:1.0638297680998221e-05, per step time: 688.7334128643604ms
2023-01-10 10:01:37,564:INFO:epoch[4], iter[1], loss:182.679425, fps:336.16 imgs/sec, lr:1.0638297680998221e-05, per step time: 761.5438243176075ms
2023-01-10 10:02:49,255:INFO:epoch[5], iter[1], loss:173.454173, fps:335.67 imgs/sec, lr:1.0638297680998221e-05, per step time: 762.66252233627ms
...
```

## 评估过程

### 评估

在运行下面的命令之前，请检查用于计算的检查点路径。文件**yolov5.ckpt **是最后保存的检查点文件，但我们将其重命名为“yolov5.ckpt”。

```shell
# run evaluation by python command
python eval.py \
    --data_dir=xxx/dataset \
    --pretrained=xxx/yolov5.ckpt \
    --eval_shape=640 > log.txt 2>&1 &
OR
# run evaluation by shell script
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

上面的python命令将在后台运行。您可以通过“log.txt”文件查看结果。测试数据集的mAP如下:

```text
# log.txt
2023-01-16 23:12:27,245:INFO:
=============coco eval result=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.147
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.364

2023-01-16 23:12:27,254:INFO:testing cost time 0.39 h
```

## 推理过程

### 导入MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

ckpt_file参数是必需的，' file_format '应该在["AIR"， "MINDIR"]中

### GPU上推理

这个inference.py文件支持GPU，batch_size可以随机设置。但是我们建议您将batch_size设置为1，以避免泄漏检测引起的错误。用法如下:

```python
# GPU inference
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

### Ascend310上推理

在执行推理之前，mindir文件必须通过' export.py '脚本导出。我们只提供了一个使用MINDIR模型进行推理的例子。

当前batch_Size只能设置为1。

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_ID]
```

* `DVPP `是必选项，必须从["DVPP"， "CPU"]中选择，不区分大小写。DVPP硬件限制宽度16对齐和高度均匀对齐。因此，网络需要使用CPU运算符来处理图像。
* `DATA_PATH`是必选项，包含图像的数据集的路径。
* `ANN_FILE`是必选项，注释文件的路径。
* `DEVICE_ID` 是可选项，默认值是0。

## 导出ONNX

* 将模型导出为ONXX：

  ```shell
  python export.py --ckpt_file /path/to/yolov5.ckpt --file_name /path/to/yolov5.onnx --file_format ONNX
  ```

## 执行ONXX评估

* 执行ONXX评估

  ```shell
  bash scripts/run_eval_onnx.sh <DATA_DIR> <ONNX_MODEL_PATH> [<DEVICE_TARGET>]
  ```

# 模型描述

## 性能

### 训练性能

| 参数          | YOLOv5s                                                  |
| ------------- | -------------------------------------------------------- |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2023-01-16                                               |
| Luojianet版本 | 1.0.6                                                    |
| 数据集        | DOTA-V1.5                                                |
| 训练参数      | epoch=320，batch_size=32，lr=0.02，momentum=0.9          |
| 优化器        | Momentum                                                 |
| 损失函数      | 带logits的Sigmoid交叉熵                                  |
| 输出          | 边界框和标签                                             |
| 损失          | 38                                                       |
| 速度          | 8卡：600-800毫秒/步;                                     |
| 总时长        | 8卡：6h47min                                             |
| 微调检查点    | 53.62M (.ckpt文件)                                       |

### 评估性能

| 参数          | YOLOv5s                                                  |
| ------------- | -------------------------------------------------------- |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2023-01-16                                               |
| Luojianet版本 | 1.0.6                                                    |
| 数据集        | DOTA-V1.5                                                |
| 训练参数      | epoch=320，batch_size=32，lr=0.02，momentum=0.9          |
| 优化器        | Momentum                                                 |
| 损失函数      | 带logits的Sigmoid交叉熵                                  |
| 输出          | 边界框和标签                                             |
| 损失          | 38                                                       |
| 速度          | 1卡: 410FPS                                              |
| 总时长        | 1卡：23min                                               |
| 微调检查点    | 53.62M (.ckpt文件)                                       |
| Map           | 57%                                                      |

# 随机情况说明

在dataset.py中，我们在“create_dataset”函数中设置了种子。我们还在train.py中使用随机种子。