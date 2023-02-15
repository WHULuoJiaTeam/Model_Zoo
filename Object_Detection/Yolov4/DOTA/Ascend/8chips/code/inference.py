# Copyright 2021, 2022, 2023 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022, 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import datetime
import time

from luojianet_ms.context import ParallelMode
from luojianet_ms import context
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.dataset.vision.c_transforms import HWC2CHW
import luojianet_ms.dataset as ds

from src.yolo import YOLOV4CspDarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.eval_utils import apply_eval
from src.transforms import reshape_fn
from src.eval_utils import Inferer

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


from PIL import Image
import cv2
import argparse

config.logger = get_logger(config.infer_log, config.rank)
def get_eval_args():
    """Get eval args"""
    parser = argparse.ArgumentParser(description='Yolov4 inference')
    parser.add_argument("--img_path", type=str, default="/media/xx/新加卷/DATA/NWPU/val_test")
    parser.add_argument("--ckpt_path", type=str, default="/home/xx/Desktop/yolov4/outputs/best_map.ckpt")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

class TestDataset:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_name = os.listdir(self.img_path)
    def __getitem__(self, item):
        one_img_name = os.path.join(self.img_path, self.img_name[item])
        img = Image.open(one_img_name).convert("RGB")
        img_id = item
        return img, img_id
    def __len__(self):
        return len(self.img_name)

def infer_img(args):
    config.per_batch_size = args.batch_size
    all_img_name = os.listdir(args.img_path)
    infer_dataset = TestDataset(args.img_path)
    hwc_to_chw = HWC2CHW()

    dataset = ds.GeneratorDataset(infer_dataset, column_names=["image", "img_id"],
                                 shuffle=False)
    compose_map_func = (lambda image, img_id: reshape_fn(image, img_id, config))
    dataset = dataset.map(operations=compose_map_func, input_columns=["image", "img_id"],
                output_columns=["image", "image_shape", "img_id"],
                column_order=["image", "image_shape", "img_id"],
                num_parallel_workers=8)
    dataset = dataset.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=8)
    dataset = dataset.batch(config.per_batch_size, drop_remainder=True)

    network = YOLOV4CspDarkNet53()
    param_dict = load_checkpoint(args.ckpt_path)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    load_param_into_net(network, param_dict_new)

    config.logger.info("Load model sucess!")

    network.set_train(False)
    detection = Inferer(config, config.ignore_threshold)
    config.logger.info('Start inference....')

    batch_num = -1

    for index, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
        batch_num = batch_num + 1
        image = data["image"]
        image_shape_ = data["image_shape"]
        image_id_ = data["img_id"]
        prediction = network(image)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id_ = image_id_.asnumpy()
        image_shape_ = image_shape_.asnumpy()
        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape_, image_id_)

        predict_result = detection.do_nms_for_results()
        img_id = 0
        all_bboxes = []
        all_classes = []
        oneimg_all_bboxes = []
        oneimg_all_classes = []

        # creat mapping relation between image and annotations
        for i, img_anno in enumerate(predict_result):
            if img_anno['image_id'] == img_id:
                oneimg_all_bboxes.append(img_anno['bbox'])
                oneimg_all_classes.append(img_anno['category_id'])
                if i == (len(predict_result)-1):
                    all_bboxes.append(oneimg_all_bboxes)
                    all_classes.append(oneimg_all_classes)
            else:
                all_bboxes.append(oneimg_all_bboxes)
                all_classes.append(oneimg_all_classes)
                img_id = img_anno['image_id']
                oneimg_all_bboxes = []
                oneimg_all_classes = []
                oneimg_all_bboxes.append(img_anno['bbox'])
                oneimg_all_classes.append(img_anno['category_id'])

        # draw annotations on image one by one
        img_index = -1
        for img_name in all_img_name[batch_num*config.per_batch_size : (batch_num+1)*config.per_batch_size]:
            img_index = img_index + 1
            result_img = cv2.imread(os.path.join(args.img_path, img_name))
            img_box = all_bboxes[img_index]
            img_class = all_classes[img_index]
            img_info = zip(img_box, img_class)
            for one_box, one_class in img_info:
                xminymin = (int(one_box[0]), int(one_box[1]))
                xmaxymax = (int(one_box[0] + one_box[2]), int(one_box[1] + one_box[3]))
                cv2.rectangle(result_img, xminymin, xmaxymax, (0,255,0), 2)
                cv2.putText(result_img, config.labels[one_class-1], xminymin, cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), thickness=2)
            cv2.imwrite(config.infer_log + '/' + img_name, result_img)
    config.logger.info('End Inference!')

if __name__ == "__main__":
    args = get_eval_args()
    infer_img(args)
