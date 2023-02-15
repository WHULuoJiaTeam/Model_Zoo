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

from osgeo import gdal
import os
import json
from tqdm import tqdm
import argparse

def write_json_dict(save_path, load_dict):
    with open(save_path, "w") as f:
        json.dump(load_dict, f)

def split(label_path, image_path, count, block=512):
    result_dict = {}
    data = gdal.Open(label_path)
    width = data.RasterXSize
    height = data.RasterYSize
    for x in range(0, width, block):
        if width < block:
            block_x = width
        else:
            block_x = width - x if block > width - x else block
        for y in range(0, height, block):
            if height < block:
                block_y = height
            else:
                block_y = height - y if block > height - y else block
            result_dict[str(count)] = {
                'imagePath': image_path,
                'labelPath': label_path,
                'x': x,
                'y': y,
                'block_x': block_x,
                'block_y': block_y,
                'width': width,
                'height': height
            }
            count += 1
    return result_dict, count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', default='./GID/Large-scale-Classification_5classes/image_RGB/')
    parser.add_argument('--label_root', default='./GID/Large-scale-Classification_5classes/label_5classes/')
    args = parser.parse_args()

    image_list = os.listdir(args.image_root)
    image_list_train = image_list[0:120]
    image_list_test = image_list[120:len(image_list)]

    count = 0
    result_all_train = {}
    for imagename in tqdm(image_list_train):
        if imagename[-4:] == '.tif':
            image_path = os.path.join(args.image_root, imagename)
            label_path = os.path.join(args.label_root, imagename.replace('.tif', '_label.tif'))
            assert os.path.exists(image_path)
            assert os.path.exists(label_path)
            result_dict, count = split(label_path, image_path, count, block=512)
            result_all_train.update(result_dict)
    write_json_dict('LCC5C_b512_woOverlap.json', result_all_train)

    count = 0
    result_all_test = {}
    for imagename in tqdm(image_list_test):
        if imagename[-4:] == '.tif':
            image_path = os.path.join(args.image_root, imagename)
            label_path = os.path.join(args.label_root, imagename.replace('.tif', '_label.tif'))
            assert os.path.exists(image_path)
            assert os.path.exists(label_path)
            result_dict, count = split(label_path, image_path, count, block=512)
            result_all_test.update(result_dict)
    write_json_dict('LCC5C_b512_woOverlap_test.json', result_all_test)
