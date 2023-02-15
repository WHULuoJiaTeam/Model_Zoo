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
import { CommonResult, ListObjectsResult, ListVersionsResult, ListMultipartResult, DeleteObjectsResult } from '../types';

/**
 * 判断桶是否存在
 * @param obsClient obs客户端，因obsClient为引入的obs库的类型，本身并未导出其类型，故使用any，下同
 * @param bucketName 桶名
 * @returns
 */
export async function hasBucket(obsClient: any, bucketName: string): Promise<boolean> {
    const promise = await obsClient.headBucket({
        Bucket: bucketName,
    });
    return !(promise.CommonMsg.Status === 404);
}

/**
 * 创建桶
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param region 桶所在region
 * @param publicRead 是否开放公共读权限
 * @param storageClass 存储类型
 * @returns
 */
export async function createBucket(
    obsClient: any,
    bucketName: string,
    region: string,
    publicRead: boolean,
    storageClass?: string
): Promise<boolean> {
    if (await hasBucket(obsClient, bucketName)) {
        core.setFailed(`The bucket: ${bucketName} already exists.`);
        return false;
    }
    obsClient
        .createBucket({
            Bucket: bucketName,
            Location: region,
            ACL: publicRead ? obsClient.enums['AclPublicRead'] : obsClient.enums['AclPrivate'],
            StorageClass: storageClass ? obsClient.enums[storageClass] : '',
        })
        .then((result: CommonResult) => {
            if (result.CommonMsg.Status < 300) {
                if (result.InterfaceResult) {
                    core.info(`create bucket: ${bucketName} Successfully.`);
                    return true;
                }
            } else {
                core.setFailed(`create bucket: ${bucketName} failed, ${result.CommonMsg.Code}.`);
                return false;
            }
        })
        .catch((err: string) => {
            core.setFailed(`create bucket: ${bucketName} failed, ${err}.`);
            return false;
        });
    return false;
}

/**
 * 获取桶的多版本状态
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @returns
 */
export async function getBucketVersioning(obsClient: any, bucketName: string): Promise<string> {
    const result = await obsClient.getBucketVersioning({
        Bucket: bucketName,
    });
    if (result.CommonMsg.Status < 300) {
        return result.InterfaceResult.VersionStatus;
    } else {
        core.info(`get bucket versioning failed because ${result.CommonMsg.Code}`);
        return '';
    }
}

/**
 * 根据前缀和起始位置，列举桶内对象
 * @param obsClient Obs客户端
 * @param bucketName 桶名
 * @param obsPath obs上请求的对象前缀
 * @param marker 起始位置
 * @returns
 */
export async function listObjects(
    obsClient: any,
    bucketName: string,
    obsPath: string,
    marker?: string
): Promise<ListObjectsResult> {
    return await obsClient.listObjects({
        Bucket: bucketName,
        Prefix: obsPath,
        Marker: marker ?? '',
    });
}

/**
 * 列举桶内全部对象
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param nextMarker 起始位置
 */
export async function getAllObjects(
    obsClient: any,
    bucketName: string,
    nextMarker?: string
): Promise<{ Key: string }[]> {
    const objList: { Key: string }[] = [];
    let isTruncated = true;
    let marker = nextMarker;

    while (isTruncated) {
        const result = await listObjects(obsClient, bucketName, '', marker);

        result.InterfaceResult.Contents.forEach((elem) => {
            objList.push({ Key: elem['Key'] });
        });
        isTruncated = result.InterfaceResult.IsTruncated === 'true';
        marker = result.InterfaceResult.NextMarker;
    }
    return objList;
}

/**
 * 根据起始位置，列举多版本桶内对象
 * @param obsClient Obs客户端
 * @param bucketName 桶名
 * @param nextKeyMarker 列举多版本对象的起始位置
 * @param nextVersionIdMarker 作为标记的多版本对象的版本, 与nextKeyMarker配合使用
 * @returns
 */
export async function listVersionObjects(
    obsClient: any,
    bucketName: string,
    nextKeyMarker?: string,
    nextVersionIdMarker?: string
): Promise<ListVersionsResult> {
    return await obsClient.listVersions({
        Bucket: bucketName,
        KeyMarker: nextKeyMarker ?? '',
        VersionIdMarker: nextVersionIdMarker ?? '',
    });
}

/**
 * 列举桶内全部多版本对象
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param nextKeyMarker 多版本对象的起始位置
 * @param nextVersionMaker 作为标记的多版本对象的版本
 */
export async function getAllVersionObjects(
    obsClient: any,
    bucketName: string,
    nextKeyMarker?: string,
    nextVersionMaker?: string
): Promise<{ Key: string; VersionId: string }[]> {
    const objList: { Key: string; VersionId: string }[] = [];
    let isTruncated = true;
    let keyMarker = nextKeyMarker;
    let versionMaker = nextVersionMaker;

    while (isTruncated) {
        const result = await listVersionObjects(obsClient, bucketName, keyMarker, versionMaker);

        result.InterfaceResult.Versions.forEach((elem) => {
            objList.push({
                Key: elem['Key'],
                VersionId: elem['VersionId'],
            });
        });
        result.InterfaceResult.DeleteMarkers.forEach((elem) => {
            objList.push({
                Key: elem['Key'],
                VersionId: elem['VersionId'],
            });
        });

        isTruncated = result.InterfaceResult.IsTruncated === 'true';
        keyMarker = result.InterfaceResult.NextKeyMarker;
        versionMaker = result.InterfaceResult.NextVersionIdMarker;
    }

    return objList;
}

/**
 * 列举桶内分段上传任务
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param nextKeyMarker 分段上传任务的起始位置
 * @param nextUploadIdMarker 指定起始位置任务的uploadid
 * @returns
 */
export async function listMultipartUploads(
    obsClient: any,
    bucketName: string,
    nextKeyMarker?: string,
    nextUploadIdMarker?: string
): Promise<ListMultipartResult> {
    return await obsClient.listMultipartUploads({
        Bucket: bucketName,
        KeyMarker: nextKeyMarker ?? '',
        UploadIdMarker: nextUploadIdMarker ?? '',
    });
}

/**
 * 列举桶内全部分段上传任务
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param nextKeyMarker 分段上传任务的起始位置
 * @param nextUploadIdMarker 起始位置任务的uploadid
 * @returns
 */
export async function getAllMultipartUploads(
    obsClient: any,
    bucketName: string,
    nextKeyMarker?: string,
    nextUploadIdMarker?: string
): Promise<{ Key: string; UploadId: string }[]> {
    const partList: { Key: string; UploadId: string }[] = [];
    let isTruncated = true;
    let keyMarker = nextKeyMarker;
    let uploadIdMarker = nextUploadIdMarker;

    while (isTruncated) {
        const result = await listMultipartUploads(obsClient, bucketName, keyMarker, uploadIdMarker);

        result.InterfaceResult.Uploads.forEach((elem) => {
            partList.push({
                Key: elem['Key'],
                UploadId: elem['UploadId'],
            });
        });

        isTruncated = result.InterfaceResult.IsTruncated === 'true';
        keyMarker = result.InterfaceResult.NextKeyMarker;
        uploadIdMarker = result.InterfaceResult.NextUploadIdMarker;
    }

    return partList;
}

/**
 * 判断桶内是否存在对象/多版本对象/任务
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @returns
 */
export async function isBucketEmpty(obsClient: any, bucketName: string): Promise<boolean> {
    const bucketVersioning = await getBucketVersioning(obsClient, bucketName);
    if (bucketVersioning === 'Enabled') {
        return (
            (await getAllVersionObjects(obsClient, bucketName)).length +
                (await getAllMultipartUploads(obsClient, bucketName)).length ===
            0
        );
    } else if (bucketVersioning === 'Suspended' || bucketVersioning === undefined) {
        return (
            (await getAllObjects(obsClient, bucketName)).length +
                (await getAllMultipartUploads(obsClient, bucketName)).length ===
            0
        );
    }
    return false;
}

/**
 * 清空桶内全部对象和任务
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @returns
 */
export async function clearBuckets(obsClient: any, bucketName: string): Promise<boolean> {
    core.info('start clear bucket');
    const clearObject = await deleteAllObjects(obsClient, bucketName);
    const clearPart = await abortAllMultipartUpload(obsClient, bucketName);
    if (clearObject && clearPart) {
        core.info(`bucket: ${bucketName} cleared successfully.`);
        return true;
    }
    return false;
}

/**
 * 删除桶内全部对象/多版本对象
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @returns
 */
export async function deleteAllObjects(obsClient: any, bucketName: string): Promise<boolean> {
    const bucketVersioning = await getBucketVersioning(obsClient, bucketName);

    let objectList: { Key: string; VersionId: string }[] | { Key: string }[] = [];
    if (bucketVersioning === 'Enabled') {
        objectList = await getAllVersionObjects(obsClient, bucketName);
    } else if (bucketVersioning === 'Suspended' || bucketVersioning === undefined) {
        objectList = await getAllObjects(obsClient, bucketName);
    } else {
        return false;
    }

    if (objectList.length === 0) {
        return true;
    }

    core.info('start clear objects.');
    // 批量删除一次仅支持最大1000个
    while (objectList.length > 1000) {
        await deleteObjects(obsClient, bucketName, objectList.splice(0, 1000));
    }
    await deleteObjects(obsClient, bucketName, objectList);

    objectList = await getAllObjects(obsClient, bucketName);
    if (objectList.length > 0) {
        core.info('delete all objects failed, please try again or delete objects by yourself.');
        return false;
    } else {
        core.info('finish clear objects.');
        return true;
    }
}

/**
 * 批量删除对象/多版本对象
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param delList 待删除对象列表
 */
async function deleteObjects(
    obsClient: any,
    bucketName: string,
    delList: { Key: string; VersionId: string }[] | { Key: string }[]
) {
    await obsClient
        .deleteObjects({
            Bucket: bucketName,
            Quiet: false,
            Objects: delList,
        })
        .then((result: DeleteObjectsResult) => {
            if (result.CommonMsg.Status === 400) {
                core.info(`Delete failed because ${result.CommonMsg.Code}`);
            } else if (result.InterfaceResult.Errors.length > 0) {
                core.info(`Failed to delete objects: ${result.InterfaceResult.Errors}.`);
            } else {
                core.info(`Successfully delete ${delList.length} objects.`);
            }
        });
}

/**
 * 取消桶内所有分段上传任务
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @returns
 */
export async function abortAllMultipartUpload(obsClient: any, bucketName: string): Promise<boolean> {
    const partList = await getAllMultipartUploads(obsClient, bucketName);
    if (partList.length === 0) {
        return true;
    }

    core.info('start clear part.');
    for (const part of partList) {
        await obsClient.abortMultipartUpload({
            Bucket: bucketName,
            Key: part.Key,
            UploadId: part.UploadId,
        });
    }
    core.info('finish clear part.');
    return true;
}

/**
 * 删除桶
 * @param obsClient obs客户端
 * @param bucketName 桶名
 * @param forceClear 是否强制清空桶
 * @returns
 */
export async function deleteBucket(obsClient: any, bucketName: string, forceClear: boolean): Promise<boolean> {

    // 若桶不存在，退出
    if (!await hasBucket(obsClient, bucketName)) {
        core.setFailed(`The bucket: ${bucketName} does not exists.`);
        return false;
    }

    // 若桶非空且用户设置不强制清空桶，退出
    let isEmpty = await isBucketEmpty(obsClient, bucketName);
    if (!isEmpty && !forceClear) {
        core.setFailed(
            'some object or parts already exist in bucket, please delete them first or set parameter "clear_bucket" as true.'
        );
        return false;
    }

    if (!isEmpty) {
        isEmpty = await clearBuckets(obsClient, bucketName);
    }

    obsClient.deleteBucket({
            Bucket: bucketName,
        })
        .then((result: CommonResult) => {
            if (result.CommonMsg.Status < 300) {
                core.info(`delete bucket: ${bucketName} successfully.`);
                return true;
            } else {
                core.setFailed(`delete bucket: ${bucketName} failed, ${result.CommonMsg.Code}.`);
            }
        })
        .catch((err: string) => {
            core.setFailed(`delete bucket: ${bucketName} failed, ${err}.`);
        });
    return false;
}
