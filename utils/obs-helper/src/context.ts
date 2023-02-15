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
import * as cred from './credential';
import { ObjectInputs, BucketInputs, CommonInputs } from './types';

export function getOperationType(): string {
    return core.getInput('operation_type', { required: true });
}

export function getCommonInputs(): CommonInputs {
    return {
        accessKey: cred.getCredential('access_key', true),
        secretKey: cred.getCredential('secret_key', true),
        endpoint: cred.getCredential('endpoint', true),
        region: core.getInput('region', { required: true }),
        operationType: core.getInput('operation_type', { required: true }),
        bucketName: core.getInput('bucket_name', { required: true }),
    };
}

export function getObjectInputs(): ObjectInputs {
    return {
        accessKey: cred.getCredential('access_key', true),
        secretKey: cred.getCredential('secret_key', true),
        endpoint: cred.getCredential('endpoint', true),
        region: core.getInput('region', { required: true }),
        operationType: core.getInput('operation_type', { required: true }),
        bucketName: core.getInput('bucket_name', { required: true }),
        localFilePath: core.getMultilineInput('local_file_path', { required: false }),
        obsFilePath: core.getInput('obs_file_path', { required: false }),
        includeSelfFolder:
            core.getBooleanInput('include_self_folder', { required: false }) ?? false,
        exclude: core.getMultilineInput('exclude', { required: false }),
    };
}

export function getBucketInputs(): BucketInputs {
    return {
        accessKey: cred.getCredential('access_key', true),
        secretKey: cred.getCredential('secret_key', true),
        endpoint: cred.getCredential('endpoint', true),
        region: core.getInput('region', { required: true }),
        operationType: core.getInput('operation_type', { required: true }),
        bucketName: core.getInput('bucket_name', { required: true }),
        publicRead: core.getBooleanInput('public_read', { required: false }),
        storageClass: core.getInput('storage_class', { required: false }),
        clearBucket: core.getBooleanInput('clear_bucket', { required: false }),
    };
}

/**
 * 根据ak/sk，初始化Obs客户端
 * @param ak AK
 * @param sk SK
 * @param server 连接OBS的服务地址
 * @returns obsClient为引入的obs库的类型，本身并未导出其类型，故使用any
 */
export function getObsClient(ak: string, sk: string, server: string): any {
    const ObsClient = require('esdk-obs-nodejs'); // eslint-disable-line @typescript-eslint/no-var-requires
    try {
        const obs = new ObsClient({
            access_key_id: ak,
            secret_access_key: sk,
            server: server,
        });
        return obs;
    } catch (error) {
        core.setFailed('init obs client fail.');
    }
}
