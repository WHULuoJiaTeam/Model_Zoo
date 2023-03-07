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

VGG是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5x5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

[Paper Link](https://arxiv.org/abs/1409.1556):
Simonyan, Karen and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” CoRR abs/1409.1556 (2014): n. pag.


## [模型架构](#目录)

![网络示意图](image.png)

## [数据集](#目录)


使用的数据集：[WHU-RS19](http://www.captain-whu.com/repository.html)  
支持的数据集：[WHU-RS19]或与WHU-RS19格式相同的数据集

- 数据集的文件目录结构如下所示
    ```text
        ├── dataset
            ├── WHU-RS19
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

## [环境要求](#目录)
本代码为华为Modelarts Ascend平台**8卡**版本

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
  # 在Ascend上训练8卡
  # （1）上传或复制预训练的模型到S3桶。
  # （2）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （3）在网站UI界面上设置代码目录为“/path/VGG”。
  # （4）在网站UI界面上设置启动文件为“train.py”。
  # （5）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （6）创建作业。
  #
  # 在Ascend上评估8卡
  # （1）上传或复制训练好的模型到S3桶。
  # （2）上传zip数据集到S3桶。 (您也可以上传源数据集，但可能很慢。)
  # （3）在网站UI界面上设置代码目录为“/path/VGG”。
  # （4）在网站UI界面上设置启动文件为“eval.py”。
  # （5）在网站UI界面上设置“数据集路径”、“输出文件路径”和“作业日志路径”。
  # （6）创建作业。
  ```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

```text
└─VGG
  ├─README.md
  ├─README_CN.md
  ├─vgg.py                          # VGG网络模型
  ├─config.py                       # 模型设置
  ├─utils.py                        # 数据读取函数，loss函数等
  ├─test.py                         # 推理网络
  ├─eval.py                         # 评估网络
  └─train.py                        # 训练网络
```

### [训练过程](#目录)

#### [脚本参数](#目录)

config.py中主要参数如下：

```
    "device_target":"Ascend",                              #GPU、CPU、Ascend
    "dataset_path": "WHU-RS19/",                           #数据存放位置
    "save_checkpoint_path": "./checkpoint",                #保存的参数存放位置
    "resume":False,                                        #是否载入模型训练
    "class_num": 19,                                       #数据集中包含的种类
    "batch_size": 4,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "epoch_size": 350,                                     #训练次数
    "save_checkpoint": True,                               #是否保存模型
    "save_checkpoint_epochs": 1,                           #多少次迭代保存一次模型
    "keep_checkpoint_max": 100,                            #文件内保存模型的最大个数，超过则删掉最开始的
    "opt": 'sgd',                                          #优化器：rmsprop或sgd
    "opt_eps": 0.001, 
    "warmup_epochs": 50,                                   #warmup训练策略
    "lr_decay_mode": "warmup",                             #学习率衰减方式：steps、poly、cosine以及warmup
    "use_label_smooth": True, 
    "label_smooth_factor": 0.1,
    "lr_init": 0.0001,                                     #初始学习率
    "lr_max": 0.001,                                       #最大学习率
    "lr_end": 0.00001                                      #最小学习率
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

训练好的模型会根据前面设置的参数保存在相应的目录下，选择合适的模型，使用prediction.py进行推理，在终端运行``python test.py --input_file **** --output_folder **** --checkpoint_path **** --classes_file ****``进行推理，其参数设置如下 

```
   --input_file, type=str, default=None, help='Input file path'
   --output_folder, type=str, default=None, help='Output file path'
   --checkpoint_path, type=str, default=None, help='Saved checkpoint file path'
   --classes_file, type=str, default=None, help='Classes saved txt path '
   --device_target, type=str, default="Ascend", help='Device target'
```

# [随机情况说明](#目录)

在eval.py中，我们设置了初始化的种子。
在test.py中，我们设置了初始化的种子。

# [ModelZoo主页](#目录)

请浏览[Model Zoo](https://github.com/WHULuoJiaTeam/Model_Zoo)。
