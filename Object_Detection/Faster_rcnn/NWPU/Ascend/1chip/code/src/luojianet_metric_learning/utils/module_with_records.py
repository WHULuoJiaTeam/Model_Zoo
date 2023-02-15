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

import luojianet_ms.nn as nn
from luojianet_ms import Tensor, Parameter
import numpy as np

from . import common_functions as c_f


class ModuleWithRecords(nn.Module):
    def __init__(self, collect_stats=True):
        super().__init__()
        self.collect_stats = collect_stats

        # self.losses_size = Parameter(int(0), requires_grad=False, name='losses_size_for_graph_mode')  # add for graph mode  requires_grad=False,

    def add_to_recordable_attributes(
        self, name=None, list_of_names=None, is_stat=False
    ):
        if is_stat and not self.collect_stats:
            pass
        else:
            c_f.add_to_recordable_attributes(
                self, name=name, list_of_names=list_of_names, is_stat=is_stat
            )
        return 0  # add for graph mode

    def reset_stats(self):
        c_f.reset_stats(self)
        return 0  # add for graph mode
