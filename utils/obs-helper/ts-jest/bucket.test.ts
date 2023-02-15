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

import { expect, test } from '@jest/globals';
import * as bucket from '../src/obs/bucket';

// --------------- base ----------------
const inputs = {
    accessKey: '******',
    secretKey: '******',
    bucketName: '******',
    region: 'cn-north-7',
    operationType: 'deleteBucket',
    clearBucket: false
};

const ObsClient = require('esdk-obs-nodejs');
const obs = new ObsClient({
    access_key_id: inputs.accessKey,       
    secret_key: inputs.secretKey,       
    server: ``
});

test('check bucket exist in bucket', () => {
    bucket.hasBucket(obs, inputs.bucketName).then((res) => {
        expect(res).toEqual(true);
    });
});

test('check get all objects in bucket', async () => {
    expect((await bucket.getAllObjects(obs, inputs.bucketName)).length).toEqual(3073);
    expect((await bucket.getAllObjects(obs, inputs.bucketName, 'over2000/spring-boot-autoconfigure')).length).toEqual(1536);
});

test('check is bucket empty', async () => {
    expect(await bucket.isBucketEmpty(obs, inputs.bucketName)).toBeFalsy();
});

test('check get bucket version status', async () => {
    expect(await bucket.getBucketVersioning(obs, inputs.bucketName)).toEqual('Suspended');
});

test('check delete objects in the bucket', async () => {
    await bucket.deleteAllObjects(obs, inputs.bucketName);
    expect(await bucket.isBucketEmpty(obs, inputs.bucketName)).toBeTruthy(); 
});
