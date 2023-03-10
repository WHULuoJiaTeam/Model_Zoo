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

try:
    import luojianet_ms
except:
    import os
    import argparse
    from PIL import Image
    import numpy as np
    from pycocotools.coco import COCO
    from luojianet_ms import context, Tensor
    from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
    import luojianet_ms.dataset as ds
    import luojianet_ms.dataset.vision.c_transforms as C
    from src.retinanet import retinanet50, resnet50, retinanetInferWithDecoder
    from src.box_utils import default_boxes
    from src.model_utils.config import config
    from src.logger import get_logger
    from src.dataset import preprocess_fn_infer
    from eval import apply_nms
    import cv2


    config.logger = get_logger(config.infer_log, config.rank)

    def get_eval_args():
        """Get eval args"""
        parser = argparse.ArgumentParser(description='RetinaNet inference')
        parser.add_argument("--img_path", type=str, default="/media/xx/新加卷/DATA/NWPU/val_test")
        parser.add_argument("--ckpt_path", type=str, default="/home/xx/Desktop/RetinaNet/ckpt/retinanet_3-270_113.ckpt")
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
            return img_id, img
        def __len__(self):
            return len(self.img_name)

    def infer_img(args):
        config.per_batch_size = args.batch_size
        all_img_name = os.listdir(args.img_path)
        infer_dataset = TestDataset(args.img_path)

        dataset = ds.GeneratorDataset(infer_dataset, column_names=["img_id", "image"], shuffle=False)
        change_swap_op = C.HWC2CHW()
        normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        compose_map_func = (lambda img_id, image: preprocess_fn_infer(img_id, image, False))
        output_columns = ["img_id", "image", "image_shape"]
        trans = [normalize_op, change_swap_op]
        dataset = dataset.map(operations=compose_map_func, input_columns=["img_id", "image"],
                    output_columns=output_columns, column_order=output_columns,
                    python_multiprocessing=False,
                    num_parallel_workers=8)
        dataset = dataset.map(operations=trans, input_columns=["image"], python_multiprocessing=False,
                    num_parallel_workers=8)
        dataset = dataset.batch(config.per_batch_size, drop_remainder=True)

        backbone = resnet50(config.num_classes)
        network = retinanet50(backbone, config)
        network = retinanetInferWithDecoder(network, Tensor(default_boxes), config)
        param_dict = load_checkpoint(args.ckpt_path)
        network.init_parameters_data()
        load_param_into_net(network, param_dict)
        network.set_train(False)

        predictions = []
        img_ids = []
        num_classes = config.num_classes
        coco_root = config.coco_root
        data_type = config.val_data_type
        val_cls = config.coco_classes
        val_cls_dict = {}
        for i, cls in enumerate(val_cls):
            val_cls_dict[i] = cls
        anno_json = os.path.join(coco_root, config.instances_set.format(data_type))
        coco_gt = COCO(anno_json)
        classs_dict = {}
        cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
        for cat in cat_ids:
            classs_dict[cat["name"]] = cat["id"]

        batch_num = -1
        for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
            batch_num = batch_num + 1
            pred_data = []
            img_id = data['img_id']
            img_np = data['image']
            image_shape = data['image_shape']

            output = network(Tensor(img_np))
            for batch_idx in range(img_np.shape[0]):
                pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                                  "box_scores": (output[1].asnumpy()[batch_idx]),
                                  "img_id": int(np.squeeze(img_id[batch_idx])),
                                  "image_shape": image_shape[batch_idx]})

            for sample in pred_data:
                pred_boxes = sample['boxes']
                box_scores = sample['box_scores']
                img_id = sample['img_id']
                h, w = sample['image_shape']

                final_boxes = []
                final_label = []
                final_score = []
                img_ids.append(img_id)

                for c in range(1, num_classes):
                    class_box_scores = box_scores[:, c]
                    score_mask = class_box_scores > config.min_score
                    class_box_scores = class_box_scores[score_mask]
                    class_boxes = pred_boxes[score_mask] * [h, w, h, w]

                    if score_mask.any():
                        # print('start nms')
                        nms_index = apply_nms(class_boxes, class_box_scores, config.nms_thershold, config.max_boxes)
                        class_boxes = class_boxes[nms_index]
                        class_box_scores = class_box_scores[nms_index]
                        final_boxes += class_boxes.tolist()
                        final_score += class_box_scores.tolist()
                        final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)
                for loc, label, score in zip(final_boxes, final_label, final_score):
                    res = {}
                    res['image_id'] = img_id
                    res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
                    res['score'] = score
                    res['category_id'] = label
                    predictions.append(res)

            img_id = 0
            all_bboxes = []
            all_classes = []
            oneimg_all_bboxes = []
            oneimg_all_classes = []

            # create mapping relation between image and annotations
            for i, img_anno in enumerate(predictions):
                if img_anno['image_id'] == img_id:
                    oneimg_all_bboxes.append(img_anno['bbox'])
                    oneimg_all_classes.append(img_anno['category_id'])
                    if i == (len(predictions) - 1):
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
            for img_name in all_img_name[batch_num * config.per_batch_size: (batch_num + 1) * config.per_batch_size]:
                img_index = img_index + 1
                result_img = cv2.imread(os.path.join(args.img_path, img_name))
                img_box = all_bboxes[img_index]
                img_class = all_classes[img_index]
                img_info = zip(img_box, img_class)
                for one_box, one_class in img_info:
                    xminymin = (int(one_box[0]), int(one_box[1]))
                    xmaxymax = (int(one_box[0] + one_box[2]), int(one_box[1] + one_box[3]))
                    cv2.rectangle(result_img, xminymin, xmaxymax, (0, 255, 0), 2)
                    cv2.putText(result_img, config.coco_classes[one_class], xminymin, cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                (255, 0, 0), thickness=2)
                cv2.imwrite(config.infer_log + '/' + img_name, result_img)

        config.logger.info('End Inference!')

    if __name__ == "__main__":
        args = get_eval_args()
        infer_img(args)
