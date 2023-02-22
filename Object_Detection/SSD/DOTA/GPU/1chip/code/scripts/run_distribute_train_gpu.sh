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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh scripts/run_distribute_train.sh [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional) [DEVICE_NUM] [RANK_SIZE]"
echo "for example: sh scripts/run_distribute_train.sh /home/neu/hrnet_final/rank_table_file_path.json coco /home/neu/ssd-coco /home/neu/coco-mindrecord .train_out /home/neu/ssdresnet34lj/resnet34.ckpt(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 6 ] && [ $# != 7 ]
then
    echo "Using: sh scripts/run_distribute_train.sh [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [DEVICE_NUM] [RANK_SIZE]"
    echo "or"
    echo "Using: sh scripts/run_distribute_train.sh [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [DEVICE_NUM] [RANK_SIZE] [PRE_TRAINED_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH1=$(get_real_path $2)    # dataset_path
PATH2=$(get_real_path $3)    # mindrecord_path
PATH3=$(get_real_path $4)    # train_output_path
PATH4=$(get_real_path $7)    # pre_trained_path
#PATH5=$(get_real_path $1)    # rank_table_file_path


if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory."
    exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: MINDRECORD_PATH=$PATH2 is not a directory."
    exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: TRAIN_OUTPUT_PATH=$PATH3 is not a directory."
fi

if [ ! -f $PATH4 ] && [ $# == 7 ]
then
    echo "error: PRE_TRAINED_PATH=$PATH4 is not a file."
    exit 1
fi

#if [ ! -f $PATH5 ]
#then
#    echo "error: RANK_TABLE_FILE_PATH=$PATH5 is not a file."
#    exit 1
#fi

export DEVICE_NUM=$5
export RANK_SIZE=$6
#export RANK_TABLE_FILE=$PATH5

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../train.py ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

if [ $# == 6 ]
then
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py --data_url $PATH1 --mindrecord_url $PATH2 --train_url $PATH3 --run_platform GPU --lr 0.075 --epoch_size 5 --dataset $1 --distribute True --device_num $DEVICE_NUM &> log.txt 2>&1 &
else
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py --data_url $PATH1 --mindrecord_url $PATH2 --train_url $PATH3 --run_platform GPU --lr 0.075 --epoch_size 5 --dataset $1 --pre_trained $PATH4 --distribute True --device_num $DEVICE_NUM &> log.txt 2>&1 &
fi
cd ..
