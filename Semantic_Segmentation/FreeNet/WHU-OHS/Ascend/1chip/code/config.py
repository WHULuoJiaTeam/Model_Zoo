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
    device_target = 'Ascend', # 设备类型，CPU,Ascend或者GPU
    dataset_path = '/cache/dataset/', # 数据集根目录，如组织格式示例中的data文件夹所在位置
    normalize = False, # 是否对影像进行归一化，False或True，若为True，则逐波段进行标准差归一化
    nodata_value = 0, # 标签中的Nodata值（即不作为样本的像素值）
    in_channels = 32, # 输入通道数（即影像波段数）
    classnum = 24, # 类别数量
    batch_size = 2, # 训练时的batchsize
    num_epochs = 100, # 训练迭代次数
    weight = None, # 是否在损失函数中对各类别加权，默认为不加权（None），若需要加权，则给出一个各类别权重的list
    learning_rate = 1e-4, # 训练学习率
    save_model_path = '/cache/checkpoint/' # 训练模型文件保存路径
)