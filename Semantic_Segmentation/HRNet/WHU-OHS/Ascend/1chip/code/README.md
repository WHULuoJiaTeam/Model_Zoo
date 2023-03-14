# Readme

## Contents

- [Readme](#readme)
  - [Contents](#contents)
  - [Model Description](#model-description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
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
HRNet, or High-Resolution Net, is a general purpose convolutional neural network for tasks like semantic segmentation, object detection and image classification. It is able to maintain high resolution representations through the whole process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams one by one, and connect the multi-resolution streams in parallel. The authors conduct repeated multi-resolution fusions by exchanging the information across the parallel streams over and over.

Link: https://doi.org/10.48550/arXiv.1908.07919

## [Model Architecture](#contents)

![Network Figure](image.png)

## [Dataset](#contents)
WHU-OHS dataset consists of about 90 million manually labeled samples of 7795 Orbita hyperspectral satellite (OHS) image patches (sized 512 × 512) from 40 Chinese locations. This dataset ranges from the visible to near-infrared range, with an average spectral resolution of 15 nm. The extensive geographical distribution, large spatial coverage, and widely used classification system make the WHU-OHS dataset a challenging benchmark. 

Download Link: http://irsip.whu.edu.cn/resources/WHU_OHS_show.php

Put the images of the data set (all images of the training set and test set) in the "image" folder, the training set label in the "train" folder, and the test set label in the "test" folder, with the name of the label the same as the name of the corresponding image. For example, organize the data in the following format:

```    
    └── data
         ├── image
         │    ├── 1.tif
         │    ├── 2.tif
         │    ├── 3.tif
         ├── train
         │    ├── 1.tif
         │    ├── 2.tif
         ├── test
         │    ├── 3.tif
```

## [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend or GPU platform.
- Framework
    - [LuojiaNet](http://58.48.42.237/luojiaNet/)
- For more information, please check the resources below：
    - [LuojiaNet tutorials](http://58.48.42.237/luojiaNet/tutorial/quickstart/)
    - [LuojiaNet Python API](http://58.48.42.237/luojiaNet/luojiaNetapi/)

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
└─HRNET
  ├── README.md
  ├── README_CN.md
  ├── config.py                     # model configuration
  ├── dataset.py                    # data loading
  ├── eval.py                       # evaluation 
  ├── model.py                      # HRNET network model
  ├── predict.py                    # prediction
  └── train.py                      # training the network
```

### [Training Process](#contents)

#### [Script Parameters](#contents)

The main parameters in ``config.py`` are as follows:

```
    device_target = 'Ascend', # device，CPU,Ascend or GPU
    dataset_path = '/cache/dataset/', # dataset root path
    normalize = False, # whether to normalize the image
    nodata_value = 0, # the Nodata value in the label
    in_channels = 32, # number of input channels (i.e. number of image bands)
    classnum = 24, # class number
    batch_size = 2, # batchsize
    num_epochs = 100, # epoch
    weight = None, # Whether to weight classes in the loss function. Default is None. If weighting is required, a list of class weights is given
    learning_rate = 1e-4, # learning rate
    save_model_path = '/cache/checkpoint/' # ckpt saving path
```
#### [Training](#contents)

Run ``python train.py`` on the terminal for training


### [Evaluation Process](#contents)

Run
```
python eval.py --dataset_path xxx --checkpoint_path xxx --device_target xxx
```
in terminal to test the network. In the command, ``--dataset_path`` indicates the root directory of the test set, ``--checkpoint_path`` indicates the trained model path, and ``--device_target ``indicates the device type, including CPU, GPU and Ascend


### [Inference Process](#contents)
Run 

```
python predict.py --input_file xxx --output_folder xxx --checkpoint_path xxx --device_target xxx
```
in terminal to interface. Where ``--input_file`` is the path of the input single image, ``--output_folder`` is the folder where the output results reside, the output result file name is the same as the input image and saved in tif format, and ``--checkpoint_path`` is the trained model path. ``--device_target`` Indicates the device type, including CPU, GPU, and Ascend

## [Description of Random Situation](#contents)

There are no random seed in our files.

## [ModelZoo Homepage](#contents)

Please check the [Model Zoo](https://github.com/WHULuoJiaTeam/Model_Zoo).
