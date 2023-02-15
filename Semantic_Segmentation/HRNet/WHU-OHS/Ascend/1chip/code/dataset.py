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

import luojianet_ms.context as context
import numpy as np
from osgeo import gdal

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# WHU-OHS数据集定义
class OHS_DatasetGenerator:
    def __init__(self, image_file_list, label_file_list, use_3D_input=False, normalize=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.normalize = normalize
        self.use_3D_input = use_3D_input

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)
        image = image_dataset.ReadAsArray().astype(np.float32)
        label = label_dataset.ReadAsArray()
        
        # 若需要对影像进行归一化，则逐波段进行标准差归一化
        if(self.normalize):
            eps = 1e-8
            bands = image.shape[0]
            image_new = np.zeros_like(image)
            for i in range(bands):
                image_i = image[i, :, :]
                mean = np.mean(image_i)
                std = np.std(image_i)
                image_new[i, :, :] = (image_i - mean) / (std + eps)

            image = image_new
        
        # 是否采用3D卷积的输入格式
        if(self.use_3D_input):
            image = image[np.newaxis, :, :, :]

        label = label.astype(np.float32) - 1

        return image, label

