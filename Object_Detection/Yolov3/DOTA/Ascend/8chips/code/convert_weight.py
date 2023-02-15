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

# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Convert weight to luojianet_ms ckpt."""
try:
    import luojianet_ms
except:
    import os
    import numpy as np
    import luojianet_ms as ms

    from src.yolo import YOLOV3DarkNet53
    from model_utils.config import config

    def load_weight(weights_file):
        """Loads pre-trained weights."""
        if not os.path.isfile(weights_file):
            raise ValueError(f'"{weights_file}" is not a valid weight file.')
        with open(weights_file, 'rb') as fp:
            np.fromfile(fp, dtype=np.int32, count=5)
            return np.fromfile(fp, dtype=np.float32)


    def build_network():
        """Build YOLOv3 network."""
        network = YOLOV3DarkNet53(is_training=True)
        params = network.get_parameters()
        params = [p for p in params if 'backbone' in p.name]
        return params


    def convert(weights_file, output_file):
        """Convert weight to luojianet_ms ckpt."""
        params = build_network()
        weights = load_weight(weights_file)
        index = 0
        param_list = []
        for i in range(0, len(params), 5):
            weight = params[i]
            mean = params[i+1]
            var = params[i+2]
            gamma = params[i+3]
            beta = params[i+4]
            beta_data = weights[index: index+beta.size].reshape(beta.shape)
            index += beta.size
            gamma_data = weights[index: index+gamma.size].reshape(gamma.shape)
            index += gamma.size
            mean_data = weights[index: index+mean.size].reshape(mean.shape)
            index += mean.size
            var_data = weights[index: index + var.size].reshape(var.shape)
            index += var.size
            weight_data = weights[index: index+weight.size].reshape(weight.shape)
            index += weight.size

            param_list.append({'name': weight.name, 'type': weight.dtype, 'shape': weight.shape,
                               'data': ms.Tensor(weight_data)})
            param_list.append({'name': mean.name, 'type': mean.dtype, 'shape': mean.shape, 'data': ms.Tensor(mean_data)})
            param_list.append({'name': var.name, 'type': var.dtype, 'shape': var.shape, 'data': ms.Tensor(var_data)})
            param_list.append({'name': gamma.name, 'type': gamma.dtype, 'shape': gamma.shape,
                               'data': ms.Tensor(gamma_data)})
            param_list.append({'name': beta.name, 'type': beta.dtype, 'shape': beta.shape, 'data': ms.Tensor(beta_data)})

        ms.save_checkpoint(param_list, output_file)


    if __name__ == "__main__":
        convert(config.input_file, config.output_file)
