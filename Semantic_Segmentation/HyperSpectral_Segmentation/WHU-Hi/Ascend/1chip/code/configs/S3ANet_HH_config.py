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
            num_classes=22,
            pool_outsize1=(944, 480),
            pool_outsize2=(472, 240),
            pool_outsize3=(236, 120),
            pool_outsize4=(118, 60)
        )
    ),


    dataset=dict(
        type='S3ANet',
        params=dict(
            train_gt_dir= "Matlab_data_format/WHU-Hi-HongHu/Training samples and test samples/Train50.mat",
            train_gt_name='HHCYtrain50',
            train_data_dir= "Matlab_data_format/WHU-Hi-HongHu/WHU_Hi_HongHu.mat",
            train_data_name='WHU_Hi_HongHu',
            encoder_size=8
        )
    ),

    test=dict(
            type='S3ANet',
            params=dict(
                test_gt_dir= "Matlab_data_format/WHU-Hi-HongHu/Training samples and test samples/Test50.mat",
                test_gt_name='HHCYtest50',
                test_data_dir= "Matlab_data_format/WHU-Hi-HongHu/WHU_Hi_HongHu.mat",
                test_data_name='WHU_Hi_HongHu',
                encoder_size=8
            )
        ),

    save_model_dir='./saved_ckpts/S3ANet_HH.ckpt',
    num_class=22,
    image_shape=(940, 475),
    picture_save_dir='./saved_ckpts/S3ANet_HH.jpg',
)