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

# from .avg_non_zero_reducer import AvgNonZeroReducer
# from .base_reducer import BaseReducer
# from .class_weighted_reducer import ClassWeightedReducer
# from .divisor_reducer import DivisorReducer
from .do_nothing_reducer import DoNothingReducer
from .mean_reducer import MeanReducer
from .multiple_reducers import MultipleReducers
# from .per_anchor_reducer import PerAnchorReducer
from .threshold_reducer import ThresholdReducer
