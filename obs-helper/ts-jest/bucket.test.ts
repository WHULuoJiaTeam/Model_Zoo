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
