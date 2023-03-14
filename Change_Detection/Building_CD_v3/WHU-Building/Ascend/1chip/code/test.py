# -*- coding: utf-8 -*-
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

"""
Created on Thu Apr 15 08:55:05 2021

@author: dell
"""

from mainnet import *
import luojianet_ms
import numpy as np
from PIL import Image
from luojianet_ms import  load_checkpoint, load_param_into_net
import os
import tqdm
import luojianet_ms.dataset.vision.py_transforms as py_vision
from luojianet_ms import context


context.set_context(device_target='GPU')

class img2tensor():
    def __init__(self):
        self.tx = py_vision.ToTensor()
    
    def forward(self, x1, x2):
        image1 = Image.open(x1)
        image2 = Image.open(x2)

        image1 = self.tx(image1)
        image2 = self.tx(image2)

        image = np.expand_dims(np.concatenate([image1, image2], 0), axis=0)

        return image

if __name__ == '__main__':
    imageval1_dir = '/home/learner/Documents/data/CD_data/test_A' # test A 时相路径
    imageval2_dir = '/home/learner/Documents/data/CD_data/test_B' # test B 时相路径

    weight_file = '/home/learner/Documents/code/BuildingChangeDetection/model/CD-85_750.ckpt' # 模型路径
    result_dir = './result' # 结果存放路径

    img2ten = img2tensor()

    net = two_net()
    param_dict = load_checkpoint(weight_file)
    load_param_into_net(net, param_dict)
    model = luojianet_ms.Model(net)

    imgs = os.listdir(imageval1_dir)

    for i, img in tqdm.tqdm(enumerate(imgs), ncols=100):
        img1 = os.path.join(imageval1_dir, img)
        img2 = os.path.join(imageval2_dir, img)

        input = img2ten.forward(img1, img2)

        output = model.predict(luojianet_ms.Tensor(input))

        output = np.array(output.squeeze(0))

        pa = output[0, :, :]
        pb = output[1, :, :]
        pd = output[2, :, :]

        # pa = ((pa > 0.5).astype('uint8'))*255
        # pa = Image.fromarray(pa)

        # pb = ((pb > 0.5).astype('uint8'))*255
        # pb = Image.fromarray(pb)

        pd = ((pd > 0.5).astype('uint8'))*255
        pd = Image.fromarray(pd)

        pd.save(os.path.join(result_dir, img))





