# 目录

- [目录](#目录)
- [GCNet说明](#GCNet说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
  - [脚本和示例代码](#脚本和示例代码)
  - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
    - [训练](#训练)
  - [评估过程](#评估过程)
    - [验证](#验证)
  - [推理过程](#推理过程)
    - [用法](#用法)
    - [结果](#结果)
- [ModelZoo主页](#modelzoo主页)

# [GCNet说明](#目录)

GC-Net (Geometry and Context Network)是一个经典的深度学习双目立体匹配模型，它以端到端的方式完成了双目密集匹配的
过程，以纠正后的核线立体像对为输入，由网络学习特征映射得到视差图，无需任何手工设计特征和后处理就能达到子像素级别的立体匹
配，大大减少工程设计的复杂性。GC-Net 网络以三维特征的形式同时考虑了图像平面特征和视差值，具有极高的鲁棒性和准确性，是
目前大多数立体匹配方法的基础架构。 

[论文](https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.html)：
Cao Y, Xu J, Lin S, et al. Gcnet: Non-local networks meet squeeze-excitation networks and beyond[C]//Proceedings of the IEEE/CVF international conference on computer vision workshops. 2019: 0-0.

# [模型架构](#目录)

![](figs/network.png)

# [数据集](#目录)

使用的数据集：[WHU-stereo](http://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html)  
支持的数据集：[WHU-stereo]或与WHU-stereo格式相同的数据集  


- 目录结构如下：

    ```text
        ├── dataset
            ├── WHU_stereo_dataset
                ├── README.txt
                ├── test_index.txt
                ├── train_index.txt
                ├── test
                │   ├─ 009_53
                │   │  ├─ Disparity
                │   │  │  ├─ 000000.png
                │   │  │  └─ ...
                │   │  ├─ Left
                │   │  │  ├─ 000000.png
                │   │  │  └─ ...
                │   │  ├─ Right
                │   │     ├─ 000000.png
                │   │     └─ ...
                │   ├── ...
                ├── train
                    ├─ 002_35
                    │  ├─ Disparity
                    │  │  ├─ 000000.png
                    │  │  └─ ...
                    │  ├─ Left
                    │  │  ├─ 000000.png
                    │  │  └─ ...
                    │  ├─ Right
                    │     ├─ 000000.png
                    │     └─ ...
                    ├── ...
    ```

建议用户使用WHU-stereo数据集来体验模型，
其他数据集需要使用与WHU-stereo相同的格式。

# [环境要求](#目录)

- 硬件 Ascend
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [LuoJiaNet](http://58.48.42.237/luojiaNet/)
- 更多关于LuojiaNet的信息，请查看以下资源：
    - [LuoJiaNet教程](https://www.luojianet.cn/tutorials/zh-CN/master/index.html)
    - [LuoJiaNet Python API](https://www.luojianet.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

- 通过官方网站安装LuoJiaNet后，您可以按照如下步骤进行训练和评估：

- 本地运行

  ```text
  # 使用python命令执行单尺度训练示例（1卡）
  python train.py --data_root=./dataset/xxx --train_list=./list/whu_training.txt --valid_list=./list/whu_validation.txt

  # 使用python命令评估
  python eval.py --data_root=./dataset/xxx --model_path=./chechpoints/xxx --eval_list=./list/whu_validation.txt
  ```

- [ModelArts](https://support.huaweicloud.com/modelarts/)上训练

  ```text
  # 在Ascend上训练1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_root='s3://dir_to_your_data'”。
  #          在base_config.yaml文件中设置"train_list='s3://dir_to_training_list/'"。
  #          在base_config.yaml文件中设置"valid_list='s3://dir_to_validation_list/'"。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在网站UI界面添加“data_root=s3://dir_to_your_data”。
  #          在网站UI界面上添加“train_list=s3://dir_to_training_list/”。
  #          在网站UI界面上添加“valid_list=s3://dir_to_validation_list/”。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制预训练的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/GCNet”。
  # （6）在网站UI界面上设置启动文件为“train.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上评估1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_root='s3://dir_to_your_data'”。
  #          在base_config.yaml文件中设置"model_path='s3://dir_to_your_trained_ckpt/'"。
  #          在base_config.yaml文件中设置"eval_list='s3://dir_to_validation_list/'"。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_root=s3://dir_to_your_data”。
  #          在base_config.yaml文件中设置"model_path=s3://dir_to_your_trained_ckpt/"。
  #          在base_config.yaml文件中设置"eval_list=s3://dir_to_validation_list/"。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制训练好的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/GCNet”。
  # （6）在网站UI界面上设置启动文件为“eval.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  #
  # 在Ascend上测试1卡
  # （1）执行a或b。
  #       a. 在base_config.yaml文件中设置“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_root='s3://dir_to_your_data'”。
  #          在base_config.yaml文件中设置"model_path='s3://dir_to_your_trained_ckpt/'"。
  #          在base_config.yaml文件中设置"eval_list='s3://dir_to_validation_list/'"。
  #          在base_config.yaml文件中设置"save_path='s3://dir_to_save_output/'"。
  #          在base_config.yaml文件中设置其他参数。
  #       b. 在网站UI界面添加“enable_modelarts=True”。
  #          在base_config.yaml文件中设置“data_root=s3://dir_to_your_data”。
  #          在base_config.yaml文件中设置"model_path=s3://dir_to_your_trained_ckpt/"。
  #          在base_config.yaml文件中设置"eval_list=s3://dir_to_validation_list/"。
  #          在base_config.yaml文件中设置"save_path=s3://dir_to_save_output/"。
  #          在网站UI界面添加其他参数。
  # （3）上传或复制训练好的模型到S3桶。
  # （4）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （5）在网站UI界面上设置代码目录为“/path/GCNet”。
  # （6）在网站UI界面上设置启动文件为“predict.py”。
  # （7）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （8）创建作业。
  ```

# [脚本说明](#目录)

## [脚本和示例代码](#目录)

```text
└─GCNet
  ├─README.md
  ├─README_CN.md
  ├─figs
    ├─network.png
    ├─result.png 
  ├─list
    ├─readme.md
    ├─whu_training.txt
    ├─whu_validation.txt
  ├─scripts
    ├─eval.sh                         # 在Ascend中启动单机验证
    ├─predict.sh                      # 在Ascend中启动单机预测
    ├─train.sh                        # 在Ascend中启动单机训练
  ├─src
    ├─dataset.py                      # 数据读取文件
    ├─data_io.py                      # 数据读写工具
    ├─benchmark_callback.py           # 中间结果生成工具
    ├─GCNet.py                        # GCNet网络模型
  ├─eval.py                           # 评估测试结果
  ├─predict.py                        # 推理端代码
  └─train.py                          # 训练网络
```

## [脚本参数](#目录)

train.py中主要参数如下：

```text
可选参数：
  --data_root           训练数据集目录
  --train_list          训练集目录
  --valid_list          验证集目录
  --crop_h              裁剪图像高度
  --crop_w              裁剪图像宽度
  --max_disp            视差最大值
  --max_h               图像最大高度
  --batch               每次训练的batch个数
  --epochs              总迭代次数
  --lr                  学习率
  --amp_level           amp水平
  --save_ckpt_epochs    保存ckpt的迭代次数
  --keep_checkpoint_max 保存ckpt的迭代次数
  --logdir              ckpt的输出路径
```

## [训练过程](#目录)

### 训练

在Ascend设备上，使用命令行语句执行单机训练示例（1卡）

```bash
sh train.sh
```
或
```
python train.py --train_list="list/whu_training.txt" --valid_list="list/whu_validation.txt" --data_root=your_dataset_path
```

上述python命令将在后台运行，您可以通过控制台查看结果。

训练结束后，您可在指定的输出文件夹下找到checkpoint文件。 得到如下损失值：

```text
INFO:epoch[1], iter[1], loss:13.840265274047852
INFO:epoch[1], iter[100], loss:7.966899394989014
INFO:epoch[1], iter[200], loss:3.101147413253784
INFO:epoch[1], iter[300], loss:1.9435290098190308
INFO:epoch[1], iter[400], loss:0.6899499297142029
INFO:epoch[1], iter[500], loss:0.55366450548172
...
```

## [评估过程](#目录)

### 评估

在LuoJiaNet环境下执行以下命令进行评估

```bash
sh eval.sh
```
或
```
python eval.py --eval_list="list/whu_validation.txt" --crop_h=384 --crop_w=768 --max_disp=160 --dataset_type="whu" --model_path="checkpoint/checkpoint_gcnet_whu-20_8316.ckpt"
```

上述python命令将在后台运行。 您可以通过控制台查看结果。 

```text
Iteration[2563|2618] name: 010_38/002009, mae(pixel): 0.09399664402008057, <1(%):0.9961039225260416, <2(%):0.9987521701388888, <3(%):0.99920654296875
Iteration[2564|2618] name: 010_38/011009, mae(pixel): 0.08075002829233806, <1(%):0.9961886935763888, <2(%):0.9990336100260416, <3(%):0.9996914333767362
Iteration[2565|2618] name: 010_38/013000, mae(pixel): 0.10973523722754584, <1(%):0.9964497884114584, <2(%):0.9978739420572916, <3(%):0.9983350965711806
Iteration[2566|2618] name: 010_38/008008, mae(pixel): 0.10264833768208821, <1(%):0.99542236328125, <2(%):0.9982401529947916, <3(%):0.9987725151909722
Iteration[2567|2618] name: 010_38/002006, mae(pixel): 0.08011961645550197, <1(%):0.9998541937934028, <2(%):0.9999932183159722, <3(%):1.0
...
```

## [推理过程](#目录)

### 用法

```shell
python predict.py --predict_list="list/whu_validation.txt" --crop_h=384 --crop_w=768 --max_disp=160 --dataset_type="whu" --model_path="checkpoint/checkpoint_gcnet_whu-20_8316.ckpt" --save_path="output/luojia_result"
```

### 结果

推理结果保存在当前路径中，您可以在log文件中找到类似如下结果。

```text
Iteration[606|2618] name: 011_38/000003, cost time:0.4417119026184082s
Iteration[607|2618] name: 011_38/012007, cost time:0.4443039894104004s
Iteration[608|2618] name: 011_38/006004, cost time:0.44126009941101074s
Iteration[609|2618] name: 011_38/008006, cost time:0.4430112838745117s
Iteration[610|2618] name: 011_38/008001, cost time:0.441241979598999s
Iteration[611|2618] name: 011_38/011003, cost time:0.44088149070739746s
Iteration[612|2618] name: 011_38/011010, cost time:0.44397759437561035s
Iteration[613|2618] name: 011_38/012005, cost time:0.4408755302429199s
Iteration[614|2618] name: 011_38/001001, cost time:0.4408762454986572s
Iteration[615|2618] name: 011_38/002002, cost time:0.44150805473327637s
Iteration[616|2618] name: 012_41/009006, cost time:0.4435269832611084s
...
```

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
