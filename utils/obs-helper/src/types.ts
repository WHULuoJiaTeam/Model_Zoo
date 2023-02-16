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

import stream from 'stream';

export const SUCCESS_STATUS_CODE = 300;

export interface CommonInputs {
    accessKey: string;
    secretKey: string;
    bucketName: string;
    operationType: string;
    endpoint: string;
    region: string;
}

export interface ObjectInputs extends CommonInputs {
    localFilePath: string[];
    obsFilePath: string;
    // 是否包含文件夹自身
    includeSelfFolder?: boolean;
    // 下载时要排除的文件夹/文件
    exclude?: string[];
}

export interface BucketInputs extends CommonInputs {
    publicRead: boolean;
    clearBucket: boolean;
    storageClass?: string;
}

export interface CommonResult {
    CommonMsg: CommonMsg;
    InterfaceResult: CommonInterfaceResult;
}

export interface CommonMsg {
    Status: number;
    Code: string;
    Message: string;
    HostId: string;
    RequestId: string;
    Id2: string;
    Indicator: string;
}

export interface CommonInterfaceResult {
    RequestId: string;
    Id2: string;
}

export interface ListObjectsResult {
    CommonMsg: CommonMsg;
    InterfaceResult: ListObjectsInterfaceResult;
}

export interface ListObjectsInterfaceResult extends CommonInterfaceResult {
    RequestId: string;
    Location: string;
    Bucket: string;
    Delimiter: string;
    IsTruncated: string;
    Prefix: string;
    Marker: string;
    NextMarker: string;
    MaxKeys: string;
    Contents: ListBucketContentItem[];
    CommonPrefixes: string[];
}

export interface ListBucketContentItem {
    ETag: string;
    Size: string;
    Key: string;
    LastModified: string;
    Owner: {
        ID: string;
    };
    StorageClass: string;
    Type: string;
}

export interface ListVersionsResult {
    CommonMsg: CommonMsg;
    InterfaceResult: ListVersionsBucketInterfaceResult;
}

export interface ListVersionsBucketInterfaceResult extends CommonInterfaceResult {
    RequestId: string;
    Location: string;
    Bucket: string;
    Delimiter: string;
    Prefix: string;
    IsTruncated: string;
    KeyMarker: string;
    VersionIdMarker: string;
    NextKeyMarker: string;
    NextVersionIdMarker: string;
    MaxKeys: string;
    Versions: VersionsItem[];
    DeleteMarkers: DeleteMarkersItem[];
    CommonPrefixes: string[];
}

export interface VersionsItem {
    ETag: string;
    Size: string;
    Key: string;
    VersionId: string;
    IsLatest: string;
    LastModified: string;
    Owner: {
        ID: string;
    };
    StorageClass: string;
    Type: string;
}

export interface DeleteMarkersItem {
    Owner: {
        ID: string;
    };
    Key: string;
    VersionId: string;
    IsLatest: string;
    LastModified: string;
}

export interface ListMultipartResult {
    CommonMsg: CommonMsg;
    InterfaceResult: ListMultipartInterfaceResult;
}

export interface ListMultipartInterfaceResult {
    RequestId: string;
    Bucket: string;
    KeyMarker: string;
    UploadIdMarker: string;
    NextKeyMarker: string;
    NextUploadIdMarker: string;
    Delimiter: string;
    Prefix: string;
    MaxUploads: string;
    IsTruncated: string;
    Uploads: PartUploads[];
    CommonPrefixes: string[];
}

export interface PartUploads {
    Key: string;
    UploadId: string;
    Initiator: {
        ID: string;
    };
    Owner: {
        ID: string;
    };
    Initiated: string;
    StorageClass: string;
}

export interface UploadObjectResult {
    CommonMsg: CommonMsg;
    InterfaceResult: UploadObjectInterfaceResult;
}

export interface UploadObjectInterfaceResult extends CommonInterfaceResult {
    RequestId: string;
    ETag: string;
    VersionId: string;
    StorageClass: string;
    SseKms: string;
    SseKmsKey: string;
    SseC: string;
    SseCKeyMd5: string;
}

export interface DownloadObjectResult {
    CommonMsg: CommonMsg;
    InterfaceResult: DownloadObjectInterfaceResult;
}

export interface DownloadObjectInterfaceResult {
    RequestId: string;
    DeleteMarker: string;
    LastModified: string;
    ContentLength: string;
    CacheControl: string;
    ContentDisposition: string;
    ContentEncoding: string;
    ContentLanguage: string;
    ContentType: string;
    Expires: string;
    ETag: string;
    VersionId: string;
    WebsiteRedirectLocation: string;
    StorageClass: string;
    Restore: string;
    AllowOrigin: string;
    AllowHeader: string;
    AllowMethod: string;
    ExposeHeader: string;
    MaxAgeSeconds: string;
    SseKms: string;
    SseKmsKey: string;
    SseC: string;
    SseCKeyMd5: string;
    Expiration: string;
    Content: string | stream.Readable;
    // 因Metadata参数官网描述类型为Object, 描述为'对象自定义元数据', 未明确说明包含数据类型，故使用any
    Metadata: any;
}

export interface UploadFileList {
    file: UploadFile[];
    folder: string[];
}

export interface UploadFile {
    local: string;
    obs: string;
}

export interface DeleteObjectsResult {
    CommonMsg: CommonMsg;
    InterfaceResult: DeleteObjectsInterfaceResult;
}

export interface DeleteObjectsInterfaceResult {
    RequestId: string;
    Deleteds: DeleteSuccess[];
    Errors: DeleteErrors[];
}

export interface DeleteSuccess {
    Key: string;
    VersionId: string;
    DeleteMarker: string;
    DeleteMarkerVersionId: string;
}

export interface DeleteErrors {
    Key: string;
    VersionId: string;
    Code: string;
    Message: string;
}
