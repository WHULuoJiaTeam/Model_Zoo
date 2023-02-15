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

import os
import random
def generate_train_list(data_root):
    dir_list = os.listdir(data_root)
    train_list = random.sample(dir_list,int(len(dir_list)*0.7))
    return train_list
if __name__ == "__main__":
    train_list = generate_train_list('D:\\eppmvsnet\\BlendedMVS')
    print(train_list)
    with open('D:\\eppmvsnet\\training_list.txt','w') as f:
        for item in train_list:
            f.write(item)
            f.write('\n')