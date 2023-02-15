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

import albumentations as A

def train_transform():
    transform = A.Compose([
        A.ToFloat(max_value=1.0),
        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5)
        ]),
        A.Normalize(mean=(0.491, 0.482, 0.447),
                    std=(0.247, 0.243, 0.262), max_pixel_value=255.0),
    ])
    return transform
def val_transform():
    transform = A.Compose([
        A.ToFloat(max_value=1.0),
        A.Normalize(mean=(0.491, 0.482, 0.447),
                    std=(0.247, 0.243, 0.262), max_pixel_value=255.0),
    ])
    return transform