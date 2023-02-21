#!/bin/bash
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

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [ANNO_FILE] [DEVICE_ID]
    DVPP is mandatory, and must choose from [DVPP|CPU], it's case-insensitive
    ANNO_PATH is mandatory, and should specify annotation file path of your data including file name.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
DVPP=${3^^}
anno=$(get_real_path $4)

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "image process mode: "$DVPP
echo "anno file: "$anno
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    if [ "$DVPP" == "DVPP" ];then
      ../ascend310_infer/out/main --mindir_path=$model --dataset_path=$data_path --device_id=$device_id --cpu_dvpp=$DVPP --aipp_path=../ascend310_infer/aipp.cfg --image_height=640 --image_width=640 &> infer.log
    elif [ "$DVPP" == "CPU"  ]; then
      ../ascend310_infer/out/main --mindir_path=$model --dataset_path=$data_path --cpu_dvpp=$DVPP --device_id=$device_id --image_height=300 --image_width=300 &> infer.log
    else
      echo "image process mode must be in [DVPP|CPU]"
      exit 1
    fi
}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --img_path=$data_path --anno_file=$anno --drop &> acc.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
