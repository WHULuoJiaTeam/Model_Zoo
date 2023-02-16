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

import logging
from skimage import io
import os
def read_txt(path):
    img_id = []
    for id in open(path):
        if len(id) > 0:
            img_id.append(id.strip())
    return img_id

class Isprs_Dataset:
    def __init__(self, img_dir, label_dir, transform, img_id_txt_path=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_id_txt_path = img_id_txt_path
        self.ids = read_txt(self.img_id_txt_path)

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i].split('/')[-1]
        img_path=os.path.join(self.img_dir, idx)
        img=io.imread(img_path)[:,:,0:3]
        label_path = os.path.join(self.label_dir, idx)
        label = io.imread(label_path)
        sample = self.transform(image=img, mask=label)
        image = sample['image'].transpose((2, 0, 1))
        label = sample['mask']
        return image, label
