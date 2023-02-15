/*
 * Copyright 2021, 2022, 2023 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022, 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

import * as core from '@actions/core';
import * as context from './context';
import * as utils from './utils';
import * as upload from './obs/upload';
import * as download from './obs/download';
import * as bucket from './obs/bucket';

async function run() {
    const commonInputs = context.getCommonInputs();

    if (!utils.checkCommonInputs(commonInputs)) {
        return;
    }

    // 初始化OBS客户端
    const obs = context.getObsClient(
        commonInputs.accessKey,
        commonInputs.secretKey,
        `${commonInputs.endpoint}`
    );

    const operationCategory = utils.getOperationCategory(context.getOperationType());

    if (operationCategory === 'object') {
        handleObject(obs);
    } else if (operationCategory === 'bucket') {
        handleBucket(obs);
    } else {
        core.setFailed(
            `please check your operation_type. you can use 'download', 'upload', 'createbucket' or 'deletebucket'.`
        );
    }
}

/**
 * 处理对象，目前支持上传对象，下载对象
 * @param obs OBS客户端
 * @returns
 */
async function handleObject(obs: any): Promise<void> {
    const inputs = context.getObjectInputs();

    if (!utils.checkObjectInputs(inputs)) {
        return;
    }

    // 若桶不存在，退出
    if (!(await bucket.hasBucket(obs, inputs.bucketName))) {
        core.setFailed(`The bucket: ${inputs.bucketName} does not exists.`);
        return;
    }

    // 执行上传/下载操作
    if (inputs.operationType === 'upload') {
        await upload.uploadFileOrFolder(obs, inputs);
    }

    if (inputs.operationType === 'download') {
        await download.downloadFileOrFolder(obs, inputs);
    }
}

/**
 * 处理桶，目前支持新增桶，删除桶
 * @param obs OBS客户端
 * @returns
 */
async function handleBucket(obs: any): Promise<void> {
    const inputs = context.getBucketInputs();

    if (!utils.checkBucketInputs(inputs)) {
        return;
    }

    if (inputs.operationType.toLowerCase() === 'createbucket') {
        bucket.createBucket(
            obs,
            inputs.bucketName,
            inputs.region,
            inputs.publicRead,
            utils.getStorageClass(inputs.storageClass ?? '')
        );
    }

    if (inputs.operationType.toLowerCase() === 'deletebucket') {
        await bucket.deleteBucket(obs, inputs.bucketName, inputs.clearBucket);
    }
}

run();
