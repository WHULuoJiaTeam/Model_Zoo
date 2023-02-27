# Contents

- [Contents](#contents)
- [YOLOv5 Description](#YOLOv5-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [CKPT](#ckpt)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#Evaluation Process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [Export ONNX](#export-onnx)
        - [Run ONNX evaluation](#run-onnx-evaluation)
- [Model Description](#model-description)
- [Performance](#performance)  
    - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)

# [YOLOv5 Description](#contents)

Published in April 2020, YOLOv5 achieved state of the art performance on the COCO dataset for object detection. It is an important improvement of YoloV3, the implementation of a new architecture in the **Backbone** and the modifications in the **Neck** have improved the **mAP**(mean Average Precision) by **10%** and the number of **FPS**(Frame per Second) by **12%**.

[code](https://github.com/ultralytics/yolov5)

# [Model Architecture](#contents)

The YOLOv5 network is mainly composed of CSP and Focus as a backbone, spatial pyramid pooling(SPP) additional module, PANet path-aggregation neck and YOLOv3 head. [CSP](https://arxiv.org/abs/1911.11929) is a novel backbone that can enhance the learning capability of CNN. The [spatial pyramid pooling](https://arxiv.org/abs/1406.4729) block is added over CSP to increase the receptive field and separate out the most significant context features. Instead of Feature pyramid networks (FPN) for object detection used in YOLOv3, the PANet is used as the method for parameter aggregation for different detector levels. To be more specifical, CSPDarknet53 contains 5 CSP modules which use the convolution **C** with kernel size k=3x3, stride s = 2x2; Within the PANet and SPP, **1x1, 5x5, 9x9, 13x13 max poolings are applied.

# [Dataset](#contents)

Dataset used: [DOTA-V1.5](<https://cocodataset.org/#download>). The image is clipped to 600*600 pixel size and 20% overlap. You can get the process code and processed dataset form here.

* Process code: https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/preprocess.zip
* Dataset（clipped）: https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/DOTA.zip

DOTA-V1.5 contains 16 common categories and 402,089 instances. Before use Yolov5 to train, please modify the dataset as coco data format. The directory structure is as follows. You can see the script description to know more. 

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

# [CKPT](#contents)

Here we give the .ckpt file(version: yolov5s). You can use this file to pretrain, eval and infer. The link is as follows:

* https://naniko.obs.cn-central-221.ovaijisuan.com/object_detection/OUTPUT/yolov5/DOTA/mult/test2_lr0.02_bs32/train/yolov5_320_94.ckpt

# [Quick Start](#contents)

After installing Luojianet via the official website, you can start training and evaluation as follows:

```bash
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

```bash
# run 1p by shell script, please change `device_target` in config file to run on Ascend/GPU, and change `T_max`, `max_epoch`, `warmup_epochs` refer to contents of notes
bash run_standalone_train.sh [DATASET_PATH]

# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

```bash
# run evaluation on Ascend/GPU by python command
python eval.py \
    --device_target="Ascend" \ # Ascend or GPU
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --pretrained="***/*.ckpt" \
    --eval_shape=640 > log.txt 2>&1 &
```

```bash
# run evaluation by shell script, please change `device_target` in config file to run on Ascend/GPU
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

Note the default_config.yaml is the default parameters for yolov5s on 8p. The `batchsize` and `lr` are different on Ascend and GPU, see the settings in `scripts/run_distribute_train.sh` or `scripts/run_distribute_train_gpu.sh`.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                              // descriptions about all the models
    ├── yolov5
        ├── README.md                          // descriptions about yolov5
        ├── scripts
        │   ├──docker_start.sh                 // shell script for docker start
        │   ├──run_distribute_train.sh         // launch distributed training(8p) in ascend
        │   ├──run_distribute_train_gpu.sh     // launch distributed training(8p) in GPU
        │   ├──run_standalone_train.sh         // launch 1p training
        │   ├──run_infer_310.sh                // shell script for evaluation on 310
        │   ├──run_eval.sh                     // shell script for evaluation
        │   ├──run_eval_onnx.sh                // shell script for onnx evaluation
        ├──model_utils
        │   ├──config.py                       // getting config parameters
        │   ├──device_adapter.py               // getting device info
        │   ├──local_adapter.py                // getting device info
        │   ├──moxing_adapter.py               // Decorator
        ├── src
        │   ├──backbone.py                     // backbone of network
        │   ├──distributed_sampler.py          // iterator of dataset
        │   ├──initializer.py                  // initializer of parameters
        │   ├──logger.py                       // log function
        │   ├──loss.py                         // loss function
        │   ├──lr_scheduler.py                 // generate learning rate
        │   ├──transforms.py                   // Preprocess data
        │   ├──util.py                         // util function
        │   ├──yolo.py                         // yolov5 network
        │   ├──yolo_dataset.py                 // create dataset for YOLOV5
        ├── default_config.yaml                // parameter configuration(yolov5s 8p)
        ├── train.py                           // training script
        ├── eval.py                            // evaluation script
        ├── eval_onnx.py                       // ONNX evaluation script
        ├── export.py                          // export script
```

## [Script Parameters](#contents)

```text
Major parameters in train.py are:

optional arguments:

  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --data_dir            Train dataset directory.
  --per_batch_size      Batch size for Training. Default: 32(1p) 16(Ascend 8p) 32(GPU 8p).
  --resume_yolov5       The ckpt file of YOLOv5, which used to fine tune.Default: ""
  --lr_scheduler        Learning rate scheduler, options: exponential,cosine_annealing.
                        Default: cosine_annealing
  --lr                  Learning rate. Default: 0.01(1p) 0.02(Ascend 8p) 0.025(GPU 8p)
  --lr_epochs           Epoch of changing of lr changing, split with ",". Default: '220,250'
  --lr_gamma            Decrease lr by a factor of exponential lr_scheduler. Default: 0.1
  --eta_min             Eta_min in cosine_annealing scheduler. Default: 0.
  --t_max               T-max in cosine_annealing scheduler. Default: 300(8p)
  --max_epoch           Max epoch num to train the model. Default: 300(8p)
  --warmup_epochs       Warmup epochs. Default: 20(8p)
  --weight_decay        Weight decay factor. Default: 0.0005
  --momentum            Momentum. Default: 0.9
  --loss_scale          Static loss scale. Default: 64
  --label_smooth        Whether to use label smooth in CE. Default:0
  --label_smooth_factor Smooth strength of original one-hot. Default: 0.1
  --log_interval        Logging interval steps. Default: 100
  --ckpt_path           Checkpoint save location. Default: outputs/
  --is_distributed      Distribute train or not, 1 for yes, 0 for no. Default: 0
  --rank                Local rank of distributed. Default: 0
  --group_size          World size of device. Default: 1
  --need_profiler       Whether use profiler. 0 for no, 1 for yes. Default: 0
  --training_shape      Fix training shape. Default: ""
  --resize_rate         Resize rate for multi-scale training. Default: 10
  --bind_cpu            Whether bind cpu when distributed training. Default: True
  --device_num          Device numbers per server. Default: 8
```

## [Training Process](#contents)

### Training

For Ascend device, standalone training can be started like this:

```shell
#run training example(1p) by python command
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

You should fine tune the params when run training 1p on GPU

The python command above will run in the background, you can view the results through the file `log.txt`.

After training, you'll get some checkpoint files under the **outputs** folder by default. The loss value will be achieved as follows:

```text
# grep "loss:" log.txt
2023-01-10 09:57:08,545:INFO:epoch[1], iter[1], loss:7776.806641, fps:0.95 imgs/sec, lr:1.0638297680998221e-05, per step time: 270274.2323875427ms
2023-01-10 09:59:21,231:INFO:epoch[2], iter[1], loss:723.615182, fps:181.37 imgs/sec, lr:1.0638297680998221e-05, per step time: 1411.4941028838462ms
2023-01-10 10:00:25,981:INFO:epoch[3], iter[1], loss:206.321586, fps:371.70 imgs/sec, lr:1.0638297680998221e-05, per step time: 688.732755945084ms
2023-01-10 10:01:37,555:INFO:epoch[4], iter[1], loss:186.602023, fps:336.22 imgs/sec, lr:1.0638297680998221e-05, per step time: 761.4078471001159ms
2023-01-10 10:02:49,252:INFO:epoch[5], iter[1], loss:179.163211, fps:335.72 imgs/sec, lr:1.0638297680998221e-05, per step time: 762.5398077863327ms
...
```

### Distributed Training

Distributed training example(8p) by shell script:

```bash
# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt(Ascend) or distribute_train/nohup.out(GPU). The loss value will be achieved as follows:

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

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation. The file **yolov5.ckpt** used in the  follow script is the last saved checkpoint file, but we renamed it to "yolov5.ckpt".

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

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

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

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`file_format` should be in ["AIR", "MINDIR"]

### Infer on GPU

This inference.py file support GPU and batch_size can be set randomly. But we suggest you set the batch_size as 1 to avoid error caused by leak detection. The usage is as follows:

```python
# GPU inference
python inference.py --img_path=[img_path] --ckpt_path=[ckpt_path] --batch_size=1
```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `DATA_PATH` is mandatory, path of the dataset containing images.
- `ANN_FILE` is mandatory, path to annotation file.
- `DEVICE_ID` is optional, default value is 0.

### [Export ONNX](#contents)

- Export your model to ONNX:  

  ```shell
  python export.py --ckpt_file /path/to/yolov5.ckpt --file_name /path/to/yolov5.onnx --file_format ONNX
  ```

### Run ONNX evaluation

- Run ONNX evaluation from yolov5 directory:

  ```shell
  bash scripts/run_eval_onnx.sh <DATA_DIR> <ONNX_MODEL_PATH> [<DEVICE_TARGET>]
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Train Performance

| Parameters                 | YOLOv5s                                          |
| -------------------------- | ------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G |
| uploaded Date              | 01/16/2023 (month/day/year)                      |
| Luojianet Version          | 1.0.6                                            |
| Dataset                    | DOTA-V1.5                                        |
| Training Parameters        | epoch=320, batch_size=32, lr=0.02,momentum=0.9   |
| Optimizer                  | Momentum                                         |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss     |
| outputs                    | boxes and label                                  |
| Loss                       | 38                                               |
| Speed                      | 8p: 600-800ms/step                               |
| Total time                 | 8p: 6h47min                                      |
| Checkpoint for Fine tuning | 53.62M (.ckpt file)                              |

### Evaluation Performance

| Parameters                 | YOLOv5s                                          |
| -------------------------- | ------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G |
| uploaded Date              | 01/16/2023 (month/day/year)                      |
| Luojianet Version          | 1.0.6                                            |
| Dataset                    | DOTA-V1.5                                        |
| Training Parameters        | epoch=320, batch_size=32, lr=0.02,momentum=0.9   |
| Optimizer                  | Momentum                                         |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss     |
| outputs                    | boxes and label                                  |
| Loss                       | 38                                               |
| Speed                      | 1p: 410FPS                                       |
| Total time                 | 1p: 23min                                        |
| Checkpoint for Fine tuning | 53.62M (.ckpt file)                              |
| Map                        | 57%                                              |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

