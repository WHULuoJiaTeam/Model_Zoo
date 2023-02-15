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

config = dict(
    model=dict(
        params=dict(
            in_channels=270,
            num_classes=9,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
            pool_outsize1=(552, 400),
            pool_outsize2=(276, 200),
            pool_outsize3=(138, 100),
            pool_outsize4=(69, 50)
        )
    ),

    dataset=dict(
        type='FreeNet',
        params=dict(
            train_gt_dir="Matlab_data_format/WHU-Hi-LongKou/Training samples and test samples/Train50.mat",
            train_gt_name='LKtrain50',
            train_data_dir="Matlab_data_format/WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
            train_data_name='WHU_Hi_LongKou',
            encoder_size=8
        )
    ),

    test=dict(
            type='FreeNet',
            params=dict(
                test_gt_dir="Matlab_data_format/WHU-Hi-LongKou/Training samples and test samples/Test50.mat",
                test_gt_name='LKtest50',
                test_data_dir="Matlab_data_format/WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
                test_data_name='WHU_Hi_LongKou',
                encoder_size=8
            )
        ),

    save_model_dir='./saved_ckpts/FreeNet_LK.ckpt',
    num_class=9,
    image_shape=(550, 400),
    picture_save_dir='./saved_ckpts/FreeNet_LK.jpg',

)