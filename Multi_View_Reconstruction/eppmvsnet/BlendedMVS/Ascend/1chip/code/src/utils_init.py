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

# Copyright 2021 Huawei Technologies Co., Ltd
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


from functools import reduce
import numpy as np
import math
import luojianet_ms.nn as nn
from luojianet_ms.common import initializer as init


def _calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d',
                  'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    if nonlinearity == 'tanh':
        return 5.0 / 3
    if nonlinearity == 'relu':
        return math.sqrt(2.0)
    if nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))

    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _assignment(arr, num):
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


def _calculate_in_and_out(arr):
    dim = len(arr.shape)
    if dim < 2:
        raise ValueError("If initialize data with xavier uniform, "
                         "the dimension of data must greater than 1.")

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out


def _select_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_in_and_out(array)
    return fan_in if mode == 'fan_in' else fan_out


class KaimingInit(init.Initializer):

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingInit, self).__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)

    def _initialize(self, arr):
        pass


class KaimingUniform(KaimingInit):

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        bound = math.sqrt(3.0) * self.gain / math.sqrt(fan)
        data = np.random.uniform(-bound, bound, arr.shape)

        _assignment(arr, data)


class KaimingNormal(KaimingInit):

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        std = self.gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)





def get_param_groups(network):
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

def get_adaptive_lr_param_groups(network):
    lr_1x_params = []
    lr_10x_params = []

    for x in network.trainable_params():
        parameter_name = x.name
        if 'fc8' in parameter_name:
            lr_10x_params.append(x)
        else:
            lr_1x_params.append(x)

    return lr_1x_params, lr_10x_params

def calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def get_conv_bias(cell):
    """Bias initializer for conv."""
    weight = init.initializer(init.HeUniform(negative_slope=math.sqrt(5)),
                                     cell.weight.shape, cell.weight.dtype).init_data()
    fan_in, _ = calculate_fan_in_and_fan_out(weight.shape)
    bound = 1 / math.sqrt(fan_in)
    return init.initializer(init.Uniform(scale=bound),
                                   cell.bias.shape, cell.bias.dtype)



class NetInitTool():

    def __init__(self, model, conv_init):
        self.model = model
        self.conv_init = conv_init

    def default_recurisive_init(self,custom_cell):
        for _, cell in custom_cell.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
                if cell.bias is not None:
                    fan_in, _ = _calculate_in_and_out(cell.weight)
                    bound = 1 / math.sqrt(fan_in)
                    cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                        cell.bias.shape,
                                                        cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
                if cell.bias is not None:
                    fan_in, _ = _calculate_in_and_out(cell.weight)
                    bound = 1 / math.sqrt(fan_in)
                    cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                        cell.bias.shape,
                                                        cell.bias.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                pass

    def custom_init_weight(self, model):
        """
        Init the weight of Conv2d,3d and BatchNorm2d,3d in the net.
        """
        for _, cell in model.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if self.conv_init == "XavierUniform":
                    cell.weight.set_data(init.initializer(init.XavierUniform(),
                                                          cell.weight.shape,
                                                          cell.weight.dtype))
                elif self.conv_init == "TruncatedNormal":
                    cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=0.21),
                                                          cell.weight.shape,
                                                          cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(get_conv_bias(cell))

            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer(1,
                                                     cell.gamma.shape,
                                                     cell.gamma.dtype))
                cell.beta.set_data(init.initializer(0,
                                                    cell.beta.shape,
                                                    cell.beta.dtype))
            elif isinstance(cell, nn.Conv3d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))

    def __init_weight(self):
        self.default_recurisive_init(self.model)
        self.custom_init_weight(self.model)