# -*- coding: utf-8 -*-
"""Inference

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/Inference.ipynb

# PyTorch Metric Learning
See the documentation [here](https://kevinmusgrave.github.io/pytorch-metric-learning/)

## Install the packages
!pip install pytorch-metric-learning
!pip install -q faiss-gpu
!git clone https://github.com/akamaster/pytorch_resnet_cifar10
"""
"""## Import the packages"""

import argparse

import luojianet_ms.ops as P
from luojianet_ms import set_seed, context

from src.luojianet_ml_tool.dataset import *
from src.luojianet_ml_tool.model import *
from src.luojianet_ml_tool.inference import get_model


def preprocess_fn(cfg, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, cfg.DATA.re_size, cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # if get image directly from DatasetGenerator object

    resize_h = cfg.DATA.AUG.resize_h
    resize_w = cfg.DATA.AUG.resize_w

    mean = tuple(x / 255.0 for x in cfg.DATA.AUG.mean)
    std = tuple(x for x in cfg.DATA.AUG.std)

    img = c_vision.Resize((resize_h, resize_w))(img)
    img = c_vision.Normalize(mean=mean, std=std)(img)
    img = c_vision.HWC2CHW()(img)

    img = luojianet_ms.Tensor.from_numpy(img)
    img = P.expand_dims(img, 0)

    return img


def ml_inference(ml_cfg, infer_img, inference_model, train_vectors, name_list, label_list, rank_k=5):
    """## Get nearest neighbors of a query"""
    cls_num = int(ml_cfg.DATA.cls_num)
    tmp_count = {}
    for i in range(cls_num):
        tmp_count[i] = 0

    # get nearest image
    indices, distances = inference_model.get_nearest_neighbors(infer_img, k=rank_k)
    nearest_labels = np.array(label_list)[indices].reshape(-1)

    nearest_imgs = train_vectors.asnumpy()[indices]
    nearest_names = np.array(name_list)[indices].reshape(-1)
    print(" ** {} nearest images".format(rank_k))
    for idx, alabel in enumerate(nearest_labels):
        print('\t  {} ====> {};'.format(idx, nearest_names[idx]))
        if alabel in tmp_count:
            tmp_count[alabel] += 1
        else:
            raise Exception('Wrong classification name!!!')

    max_cls = max(tmp_count.values())
    pred = list(tmp_count.keys())[list(tmp_count.values()).index(max_cls)]

    return pred


if __name__ == "__main__":
    """ cmd
    python ml_eval.py \
        --gpu_id=3 \
        --config_path=./configs/ml_standard.yaml \
        --checkpoint_path=./output_dir/ml_mini_dataset/gt_det_12-10_5.ckpt
    """

    """set random seed"""
    set_seed(1)

    # # get config file
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", required=True, type=str, default='./configs/ml_standard.yaml',
                        help="Config file path")
    parser.add_argument("--checkpoint_path", required=True, type=str, default="./output_dir/ml_mini_dataset/gt_det_12-10_5.ckpt",
                        help="Config file path")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="Config file path")

    os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().gpu_id

    """get config file"""
    cfg_path = parser.parse_args().config_path
    checkpoint_path = parser.parse_args().checkpoint_path
    # ===============================================================================================================

    # get config file
    cfg, inference_model, train_vectors, name_list, label_list, labels_to_indices = get_model(
        cfg_path=cfg_path, checkpoint_path=checkpoint_path)

    anno_list = []
    with open(os.path.join(cfg.DATA.test_root, '../', cfg.DATA.test_list), 'r') as f:  #test
        lines = f.readlines()
        for line in lines:
            if '\n' in line:
                line = line[:-1]  # the txt should have a blank line with \n
            else:
                line = line
            cur_pair = line.split(' ')
            img_name = str(cur_pair[0])
            cls = str(cur_pair[1])
            anno_tmp = [img_name, str(cls)]
            anno_list.append(anno_tmp)
    print('in data_loader: Train data preparation done')

    tp = 0
    fp = 0
    for idx, (aimg_name, alabel) in enumerate(anno_list):
        print('Processing: {} / {}, {}'.format(idx, len(anno_list), aimg_name))
        test_img_path = os.path.join(cfg.DATA.test_root, aimg_name)
        test_img = preprocess_fn(cfg, test_img_path)

        pred = ml_inference(cfg, test_img, inference_model, train_vectors, name_list, label_list, rank_k=cfg.TEST.rank_k)

        if pred == int(alabel):
            tp = tp + 1
        else:
            fp = fp + 1

    print('Precision: {}/{} = {}'.format(tp, tp+fp, tp/(tp+fp)))


