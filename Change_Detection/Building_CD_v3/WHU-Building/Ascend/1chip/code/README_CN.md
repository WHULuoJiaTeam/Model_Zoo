# Readme
## 目录

- [Readme](#readme)
  - [目录](#目录)
  - [模型说明](#模型说明)
  - [模型架构](#模型架构)
  - [数据集](#数据集)
  - [环境要求](#环境要求)
  - [快速入门](#快速入门)
  - [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [训练过程](#训练过程)
      - [脚本参数](#脚本参数)
      - [训练](#训练)
    - [评估过程](#评估过程)
    - [推理过程](#推理过程)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

## [模型说明](#目录)

Building_CD是一种基于迁移学习思想的新型多任务网络，通过适当选择高维特征进行共享和唯一的解码模块，减少了对变化检测样本的依赖。与其他多任务变化检测网络不同的是，在高精度建筑掩码的帮助下，该网络可以充分利用来自建筑检测分支的先验信息，并通过所提出的对象级优化算法进一步提高变化检测结果。

[Paper Link](https://doi.org/10.3390/rs14040957):
S. Gao, W. Li, K. Sun, J. Wei, Y. Chen, and X. Wang, “Built-Up Area Change Detection Using Multi-Task Network with Object-Level Refinement,” Remote Sensing, vol. 14, no. 4, p. 957, Feb. 2022, doi: 10.3390/rs14040957.

代码提供者：[gaosong@whu.edu.cn](gaosong@whu.edu.cn)  

## [模型架构](#目录)

![网络示意图](image.png)

## [数据集](#目录)


1. 下载遥感建筑物变化检测影像数据集，推荐使用WHU Building change detection dataset
  本例中使用的数据集：[WHU Building change detection dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 
1. 将数据集整理成如下格式：  
注意：A、B、building_A、building_B、label中的图片名字必须对应！  
其中，A，building_A分别为A时相影像以及对应的建筑物掩膜，B，building_B分别为B时相影像以及对应的建筑物掩膜，label为对应的变化掩膜

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

## [环境要求](#目录)
本代码为华为Modelarts Ascend平台**单卡**版本

- 硬件 Ascend
    - 使用 Ascend处理器 准备硬件环境。
- 框架
    - [LuojiaNet](http://58.48.42.237/luojiaNet/)
- 更多关于LuojiaNet的信息，请查看以下资源：
    - [LuojiaNet Tutorials](http://58.48.42.237/luojiaNet/tutorial/quickstart/)
    - [LuojiaNet Python API](http://58.48.42.237/luojiaNet/luojiaNetapi/)

## [快速入门](#目录)

- 通过官方网站安装LuoJiaNet后，您可以按照如下步骤进行训练和评估：


- [ModelArts](https://support.huaweicloud.com/modelarts/)上训练

  ```text
  # 在Ascend上训练1卡
  # （1）上传或复制预训练的模型到S3桶。
  # （2）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （3）在网站UI界面上设置代码目录为“/path/Building_CD”。
  # （4）在网站UI界面上设置启动文件为“train.py”。
  # （5）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （6）创建作业。
  #
  # 在Ascend上评估1卡
  # （1）上传或复制训练好的模型到S3桶。
  # （2）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （3）在网站UI界面上设置代码目录为“/path/Building_CD”。
  # （4）在网站UI界面上设置启动文件为“prediciton.py”。
  # （5）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （6）创建作业。
  ```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

```text
└─Building_CD
  ├─README.md
  ├─README_CN.md
  ├─dataset.py                      # 数据读取文件
  ├─fpn.py                          # FPN网络模型
  ├─vgg.py                          # VGG网络模型
  ├─mainet.py                       # Building_CD网络模型主干部分
  ├─finalnet.py                     # Building_CD网络模型最终部分
  ├─test.py                         # 测试结果
  ├─eval.py                         # 评估结果
  ├─prediction.py                   # 模型推理
  ├─config.py                       # 模型设置
  └─train.py                        # 训练网络
```

### [训练过程](#目录)

#### [脚本参数](#目录)

config.py中主要参数如下：

```
    "device_target":"Ascend",      #GPU、CPU、Ascend
    "device_id":0,   #显卡ID
    "dataset_path": "./CD_data",  #数据存放位置
    "save_checkpoint_path": "./checkpoint",  #保存的参数存放位置
    "resume":False,   #是否载入模型训练
    "batch_size": 8,
    "aug" : True,
    "step_per_epoch": 200,
    "epoch_size": 200, #训练次数
    "save_checkpoint": True, #是否保存模型
    "save_checkpoint_epochs": 200, #多少次迭代保存一次模型
    "keep_checkpoint_max": 5, #保存模型的最大个数
    "decay_epochs": 20, #学习率衰减的epoch数
    "max_lr": 0.001, #最大学习率
    "min_lr": 0.00001, #最小学习率
    "LR":1e-4
```

#### [训练](#目录)
在终端运行``python train.py``进行训练

### [评估过程](#目录)

训练好的模型会根据前面设置的参数保存在相应的目录下，选择合适的模型，使用eval.py进行评估，在终端运行``python eval.py --checkpoint_path **** --dataset_path ****``进行评估，其参数设置如下  

```
    --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
    --dataset_path, type=str, default=None, help='Eval dataset path'
    --device_target, type=str, default=config.device_target, help='Device target'
    --device_id, type=int, default=config.device_id, help='Device id'
```

### [推理过程](#目录)

训练好的模型会根据前面设置的参数保存在相应的目录下，选择合适的模型，使用prediction.py进行推理，在终端运行``python prediction.py --checkpoint_path **** --dataset_path ****``或``python prediction.py --checkpoint_path **** --left_input_file **** --right_input_file ****``进行推理，其参数设置如下 

```
    --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
    --dataset_path, type=str, default=None, help='Predict dataset path'
    --left_input_file, type=str, default=None, help='Pre-period image'
    --right_input_file, type=str, default=None, help='Post-period image'
    --output_folder, type=str, default="./result", help='Results path'
    --device_target, type=str, default=config.device_target, help='Device target'
    --device_id, type=int, default=config.device_id, help='Device id'
```

# [随机情况说明](#目录)

在eval.py中，我们设置了初始化的种子。
在prediction.py中，我们设置了初始化的种子。

# [ModelZoo主页](#目录)

请浏览[Model Zoo](https://github.com/WHULuoJiaTeam/Model_Zoo)。
