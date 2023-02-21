# Copyright 2022 Huawei Technologies Co., Ltd
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
import onnxruntime
import mindspore as ms
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.util import DetectionEngine
from model_utils.config import config

def conver_testing_shape(args):
    """Convert testing shape to list."""
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape


def run_test():
    """The function of eval."""
    start_time = time.time()
    config.data_root = os.path.join(config.data_dir, 'val2014')
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2014.json')

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=devid)

    # logger
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    config.logger = get_logger(config.outputs_dir, rank_id)

    ms.reset_auto_parallel_context()
    parallel_mode = ms.ParallelMode.STAND_ALONE
    ms.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)
    print(config.device_target)
    if config.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif config.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {config.device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = onnxruntime.InferenceSession(config.pretrained, providers=providers)
    input_names = [x.name for x in session.get_inputs()]

    if config.testing_shape:
        config.test_img_shape = conver_testing_shape(config)

    ds = create_yolo_dataset(config.data_root, config.annFile, is_training=False,
                             batch_size=1, device_num=1,
                             rank=rank_id, shuffle=False, config=config)

    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('totol %d images to eval', ds.get_dataset_size() * 1)
    # init detection engine
    detection = DetectionEngine(config)

    config.logger.info('Start inference....')
    for i, data in enumerate(ds.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image = data["image"]
        image_shape = data["image_shape"]
        image_id = data["img_id"]
        output = session.run(None, {input_names[0]: image})
        output_big, output_me, output_small = output
        detection.detect([output_big, output_me, output_small], 1, image_shape, image_id)
        if i % 50 == 0:
            config.logger.info('Processing... {:.2f}% '.format(i / ds.get_dataset_size() * 100))

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    config.logger.info('result file path: %s', result_file_path)
    eval_result = detection.get_eval_result()

    cost_time = time.time() - start_time
    eval_print_str = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_print_str)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)


if __name__ == "__main__":
    run_test()
