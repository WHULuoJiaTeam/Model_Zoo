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

# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
# try:
#     import luojianet_ms
# except:
import os
import time
import numpy as np

import luojianet_ms.common.dtype as mstype
from luojianet_ms import context, Tensor
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.common import set_seed, Parameter
from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.utils import bbox2result_1image
from src.luojia_detection.detectors import FasterRcnn_Infer

import cv2
import glob

def imnormalize_img(img, img_shape):
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace
    img_data = img_data.astype(np.float32)
    return img_data, img_shape

def transpose_img(img, img_shape):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    return img_data, img_shape

def preprocess_img(image):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, image_shape):
        image_shape = image_shape[:2]
        image_shape = np.append(image_shape, (1.0, 1.0))

        image_data, image_shape = imnormalize_img(image_bgr, image_shape)

        image_data, image_shape = transpose_img(image_data, image_shape)
        return image_data, image_shape

    def _data_aug(image):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]

        return _infer_data(image_bgr, image_shape)

    return _data_aug(image)

def compute_IOU(rec1, rec2):  # cur_box, abox
    """
    Compute IoU of two boxes
    :param rec1: (x0,y0,x1,y1)   - xmin, ymin, xmax, ymax
    :param rec2: (x0,y0,x1,y1)
    :return: IOU number.
    """
    left_column_max  = max(rec1[0], rec2[0])  # 1213
    right_column_min = min(rec1[2], rec2[2])  # 1227
    up_row_max       = max(rec1[1], rec2[1])  # 422
    down_row_min     = min(rec1[3], rec2[3])  # 437
    # no overlap
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0, 0,  -1
    # overlap
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)

        return S_cross / S1, S_cross / S2, (S1 >= S2)


def fasterrcnn_inference(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))
    img_paths = glob.glob(config.inference_img_dir + '*.jpg')
    net = FasterRcnn_Infer(config)

    param_dict = load_checkpoint(ckpt_path)
    if config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    for oldkey in list(param_dict.keys()):
        if oldkey.startswith(("backbone", "rcnn", "fpn_neck", "rpn_with_loss")):
            data = param_dict.pop(oldkey)
            newkey = 'network.' + oldkey
            param_dict[newkey] = data
    load_param_into_net(net, param_dict)

    net.set_train(False)
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    total = len(img_paths)
    total_time = 0
    total_num = 0
    cls_names = config.coco_classes

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128

    if not os.path.exists(config.inference_save_dir):
        os.mkdir(config.inference_save_dir)

    hbb_dir = os.path.join(config.inference_save_dir, 'hbb_result/')
    if not os.path.exists(hbb_dir):
        os.mkdir(hbb_dir)

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)  # h, w, c
        img_name = img_path.split("/")[-1]
        print("{} / {}, img_name: {}".format(idx + 1, total, img_name))
        # slide window det
        chip_size_h = config.img_height
        chip_size_w = config.img_width
        slide_size_h = chip_size_h // 2
        slide_size_w = chip_size_w // 2
        height, width, channel = img.shape
        if height < chip_size_h or width < chip_size_w:
            right_padding = max(chip_size_w - width, 0)
            down_padding = max(chip_size_h - height, 0)
            img = cv2.copyMakeBorder(img, 0, down_padding, 0, right_padding, cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
            height, width, channel = img.shape

        h_stepnum = int((height - chip_size_h) / slide_size_h) + 1
        w_stepnum = int((width - chip_size_w) / slide_size_w) + 1
        if (height - (h_stepnum - 1) * slide_size_h - chip_size_h > 0):
            down_padding = slide_size_h - (height - (h_stepnum - 1) * slide_size_h - chip_size_h)
            img = cv2.copyMakeBorder(img, 0, down_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            h_stepnum += 1
        if (width - (w_stepnum - 1) * slide_size_w - chip_size_w > 0):
            right_padding = slide_size_w - (width - (w_stepnum - 1) * slide_size_w - chip_size_w)
            img = cv2.copyMakeBorder(img, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            w_stepnum += 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # c, h, w
        img, img_metas = preprocess_img(img)
        height, width = img.shape[1], img.shape[2]
        img_metas[0], img_metas[1] = chip_size_h, chip_size_w

        img = np.expand_dims(img, axis=0)
        img_metas = np.expand_dims(img_metas, axis=0)
        img_metas = Tensor(img_metas, mstype.float32)

        flt_all_bbox = [[] for i in range(len(cls_names) - 1)]
        for i in range(w_stepnum):
            for j in range(h_stepnum):
                sub_img_data = img[:, :, j * slide_size_h:j * slide_size_h + chip_size_h,
                               i * slide_size_w:i * slide_size_w + chip_size_w]
                sub_img_data = Tensor(sub_img_data, mstype.float32)
                start = time.time()
                output = net(sub_img_data, img_metas)
                end = time.time()
                total_time += (end - start)
                # output
                all_bbox = output[0]
                all_label = output[1]
                all_mask = output[2]

                all_bbox_squee = np.squeeze(all_bbox.asnumpy()[0, :, :])
                all_label_squee = np.squeeze(all_label.asnumpy()[0, :, :])
                all_mask_squee = np.squeeze(all_mask.asnumpy()[0, :, :])

                all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
                all_labels_tmp_mask = all_label_squee[all_mask_squee]

                if all_bboxes_tmp_mask.shape[0] > max_num:
                    inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                    inds = inds[:max_num]
                    all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                    all_labels_tmp_mask = all_labels_tmp_mask[inds]

                flt_bbox = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
                for idx in range(len(cls_names) - 1):
                    if len(flt_bbox[idx]) != 0:
                        for bbox in flt_bbox[idx]:
                            if bbox[4] < 0.7:
                                continue
                            bbox[0] += i * slide_size_w
                            bbox[2] += i * slide_size_w
                            bbox[1] += j * slide_size_h
                            bbox[3] += j * slide_size_h
                            cur_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), round(bbox[4], 3)]

                            if len(flt_all_bbox[idx]) != 0:
                                box_id = []
                                islarger_id = []
                                for id_num, abbox in enumerate(flt_all_bbox[idx]):
                                    s1_iou, s2_iou, s1lrs2 = compute_IOU(cur_bbox, abbox)
                                    if s2_iou >= 0.9:
                                        box_id.append(id_num)
                                        islarger_id.append(1)
                                    elif s1_iou > 0.6:  # 0.6
                                        box_id.append(id_num)
                                        islarger_id.append(s1lrs2)

                                if len(box_id) != 0:
                                    for k in range(len(box_id)):
                                        if islarger_id[k]:
                                            flt_all_bbox[idx][box_id[k]] = cur_bbox

                                        elif not islarger_id[k]:
                                            pass
                                else:
                                    flt_all_bbox[idx].append(cur_bbox)

                            else:
                                flt_all_bbox[idx].append(cur_bbox)

        ### save inference results
        img = cv2.imread(img_path)
        # img_shape = img.shape[:2]
        img_name = img_path.split("/")[-1]
        save_path = os.path.join(config.inference_save_dir, img_name)
        cnt = 0

        for idx in range(len(cls_names) - 1):
            txt_file_name = str(idx) + '.txt'
            txt_file = os.path.join(hbb_dir, txt_file_name)
            if len(flt_all_bbox[idx]) != 0:
                for bbox in flt_all_bbox[idx]:
                    cnt += 1
                    cls_name = cls_names[idx + 1]
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, '{:.2}, {}'.format(bbox[4], cls_name), (int(bbox[0]), int(bbox[1])), font, 0.4, (0, 0, 255), thickness=2)

                    bbox_result = ' '
                    for i in bbox[:4]:
                        bbox_result += str(int(i))
                        bbox_result += ' '
                    with open(txt_file, 'a') as f:
                        f.write(img_name + ' ' + str(round(bbox[4], 3)) + bbox_result + '\n')
                    f.close()

        cv2.imwrite(save_path, img)
        total_num += cnt
        print(img_name, ", det num: ", cnt)
    print('det total num: ', total_num, ' time per image: ', total_time / total)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def inference_fasterrcnn():
    print("Start Inference!")
    fasterrcnn_inference(config.inference_checkpoint_path)  # change config-param names


if __name__ == '__main__':
    """
    python inference_fasterrcnn.py  --enable_infer \
                                    --config_path=./configs/faster_rcnn_r152_fpn.yaml \
                                    --infer_img_dir=./examples/inference_images/ \
                                    --infer_save_dir=./output_dir/inference_results/ \
                                    --infer_checkpoint_path=./pretrained_models/faster_rcnn-30_2253.ckpt
    """
    # set random seed
    set_seed(1)
    # set environment parameters
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=int(config.device_id))
    # run inference
    inference_fasterrcnn()
