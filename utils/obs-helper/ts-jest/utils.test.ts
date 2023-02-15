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

import * as fs from 'fs';
import * as utils from '../src/utils';
import { expect, test } from '@jest/globals';

// 检查region
test('test is region legal', () => {
    const result1 = utils.checkRegion('cn-north-4');
    expect(result1).toBeTruthy();

    const result2 = utils.checkRegion('cn-north-5');
    expect(result2).toBeFalsy();

    const result3 = utils.checkRegion('cn-north-');
    expect(result3).toBeFalsy();
});

// 检查桶名
test('test bucketName', ()=> {
   expect(utils.checkBucketName('aa')).toBeFalsy(); // 长度小于3
   expect(utils.checkBucketName('aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffgggg')).toBeFalsy(); // 长度大于63
   expect(utils.checkBucketName('a..a')).toBeFalsy(); // 连续的.
   expect(utils.checkBucketName('a.-a')).toBeFalsy(); // 连续的.-
   expect(utils.checkBucketName('a-.a')).toBeFalsy(); // 连续的-.
   expect(utils.checkBucketName('aA.a')).toBeFalsy(); // 有大写字母
   expect(utils.checkBucketName('a*.a')).toBeFalsy(); // 有不允许的字符
   expect(utils.checkBucketName('255.255.255.0')).toBeFalsy(); // 类IP地址
   expect(utils.checkBucketName('a.ab--c')).toBeTruthy();
   expect(utils.checkBucketName('a-----a')).toBeTruthy();  
   expect(utils.checkBucketName('88.255.256.0')).toBeTruthy();  
})

// 检查ak/sk
 test('test ak/sk', () => {
   const input1 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1/',
      localFilePath: ['resource/bigFile.zip'],
      region: 'cn-north-6',
   }
   const legal1 = utils.checkAkSk(input1.accessKey, input1.secretKey);
   expect(legal1).toBeTruthy();

   const input2 = {
      accessKey: 'KQC3',
      secretKey: '******11111222222233333444444',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1/',
      localFilePath: ['resource/bigFile.zip'],
      region: 'cn-north-6',
   }
   const legal2 = utils.checkAkSk(input2.accessKey, input2.secretKey);
   expect(legal2).toBeFalsy();
});

test('test getOperationCategory', () => {
   expect(utils.getOperationCategory('creatEBUCKet')).toEqual('bucket');
   expect(utils.getOperationCategory('DELETEbucket')).toEqual('bucket');
   expect(utils.getOperationCategory('delete')).toEqual('');
   expect(utils.getOperationCategory('UPLOAD')).toEqual('object');
   expect(utils.getOperationCategory('download')).toEqual('object');
   expect(utils.getOperationCategory('downloadfile')).toEqual('');
});

// 检查上传时的localFilePath和obsFilePath是否合法
test('check localFilePath and obsFilePath when upload', () => {
   const input1 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1',
      localFilePath: ['resource/uploadDir'],
      region: 'cn-north-6',
   };
   expect(utils.checkUploadFilePath(input1)).toBeTruthy();
   const input2 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1',
      localFilePath: ['resource/bigFile.zip', 'resource/a.txt'],
      region: 'cn-north-6',
   };
   expect(utils.checkUploadFilePath(input2)).toBeFalsy();
   const input3 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1',
      localFilePath: [],
      region: 'cn-north-6',
   };
   expect(utils.checkUploadFilePath(input3)).toBeFalsy();
   const input4 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1',
      localFilePath: [''],
      region: 'cn-north-6',
   };
   expect(utils.checkUploadFilePath(input4)).toBeFalsy();
   const input5 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'upload',
      obsFilePath: 'uploadtest1',
      localFilePath: ['path1','path2','path3','path4','path5','path6','path7','path8','path9','path10','path11'],
      region: 'cn-north-6',
   };
   expect(utils.checkUploadFilePath(input5)).toBeFalsy();
});

// 检查下载时的localFilePath和obsFilePath是否合法
test('check localFilePath and obsFilePath when download', () => {
   const input1 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'download',
      obsFilePath: 'uploadtest1',
      localFilePath: ['resource/bigFile.zip'],
      region: 'cn-north-6',
   };
   expect(utils.checkDownloadFilePath(input1)).toBeTruthy();
   const input2 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'download',
      obsFilePath: 'uploadtest1',
      localFilePath: ['resource/bigFile.zip', 'resource/a.txt'],
      region: 'cn-north-6',
   };
   expect(utils.checkDownloadFilePath(input2)).toBeFalsy();
   const input3 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'download',
      obsFilePath: 'uploadtest1',
      localFilePath: [],
      region: 'cn-north-6',
   };
   expect(utils.checkDownloadFilePath(input3)).toBeFalsy();
   const input4 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'download',
      obsFilePath: 'uploadtest1',
      localFilePath: [''],
      region: 'cn-north-6',
   };
   expect(utils.checkDownloadFilePath(input4)).toBeFalsy();
   const input5 = {
      accessKey: '******',
      secretKey: '******',
      bucketName: '******',
      operationType: 'download',
      obsFilePath: '',
      localFilePath: ['a/b'],
      region: 'cn-north-6',
   };
   expect(utils.checkDownloadFilePath(input5)).toBeFalsy();
});

test('test replace \\ to /', () => {
   const path1 = utils.replaceSlash('a\\b\\\\c');
   expect(path1).toEqual('a/b//c');

   const path2 = utils.replaceSlash('a\\c\\\\');
   expect(path2).toEqual('a/c//');

   const path3 = utils.replaceSlash('a/b/c');
   expect(path3).toEqual('a/b/c');

   const path4 = utils.replaceSlash('a/b\\\\c/');
   expect(path4).toEqual('a/b//c/');
});

test('test del first rootPath in path', () => {
   const result1 = utils.getPathWithoutRootPath('src/src1/src1-1','src/src1/src1-1/src/src1/src1-1/test.txt')
   expect(result1).toEqual('/src/src1/src1-1/test.txt');
   
   const result2 = utils.getPathWithoutRootPath('src/','src/src1/src1-1/src/src1/src1-1/test.txt')
   expect(result2).toEqual('src1/src1-1/src/src1/src1-1/test.txt');
   
   const result3 = utils.getPathWithoutRootPath('src/te','src/test.txt')
   expect(result3).toEqual('st.txt');
   
   const result4 = utils.getPathWithoutRootPath('src/tast.txt','src/test.txt')
   expect(result4).toEqual('src/test.txt');
   
   const result5 = utils.getPathWithoutRootPath('crs','src/test.txt')
   expect(result5).toEqual('src/test.txt');
});

test('test path is end with "/" or not', () => {
   const path1 = utils.isEndWithSlash('a/');
   expect(path1).toBeTruthy();

   const path2 = utils.isEndWithSlash('a/b.');
   expect(path2).toBeFalsy();

   const path3 = utils.isEndWithSlash('');
   expect(path3).toBeFalsy();
});

test('test delete "/" in the end of the path if "/" exist', () => {
   const path1 = utils.getStringDelLastSlash('a/');
   expect(path1).toEqual('a');

   const path2 = utils.getStringDelLastSlash('a/b//');
   expect(path2).toEqual('a/b/');

   const path3 = utils.getStringDelLastSlash('a/b.');
   expect(path3).toEqual('a/b.');

   const path4 = utils.getStringDelLastSlash('');
   expect(path4).toEqual('');
});

// 创建文件夹
test('test create folder', () => {
  const folder1 = utils.createFolder('resource/newFolder');
  expect(fs.existsSync('resource/newFolder')).toBeTruthy();
});

// 检查文件是否超过5GB
test('test file is oversized', () => {
  const over1 = utils.isFileOverSize('resource/bigFile.zip');
  expect(over1).toBeTruthy();

  const over2 = utils.isFileOverSize('resource/uploadDir/file1.txt');
  expect(over2).toBeFalsy();
});

// 检查本地是否存在同名文件
test('isExistSameNameFile', () => {
   expect(utils.isExistSameNameFile('resource/uploadDir/file1.txt')).toBeTruthy();
   expect(utils.isExistSameNameFile('resource/uploadDir/file2.txt')).toBeFalsy();
});

// 检查本地是否存在同名文件夹
test('isExistSameNameFolder', () => {
   expect(utils.isExistSameNameFolder('resource/uploadDir/folder1')).toBeTruthy();
   expect(utils.isExistSameNameFolder('resource/uploadDir/folder3')).toBeFalsy();
});