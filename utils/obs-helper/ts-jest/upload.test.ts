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
import * as upload from '../src/obs/upload';
import * as download from '../src/obs/download';
import { ObjectInputs } from '../src/types';

const ObsClient = require('esdk-obs-nodejs');
function getObsClient(inputs: ObjectInputs) {
    return new ObsClient({
        access_key_id: inputs.accessKey,       
        secret_access_key: inputs.secretKey,       
        server: ``
    });
}
const inputs = {
    accessKey: '******',
    secretKey: '*****************',
    bucketName: '******',
    operationType: 'upload',
    obsFilePath: '',
    localFilePath: [''],
    region: 'cn-north-7',
    includeSelfFolder: false
}

// ------------------------------file---------------------------------

test('upload a exist file without rename to obs folder "obsTest1"', async () => {
    inputs.localFilePath = ['resource/uploadDir/file1.txt'];
    inputs.obsFilePath = 'obsTest1/file1.txt'
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('obsTest1/file1.txt')).toBeGreaterThan(-1);
});

test('upload a exist file and rename to obs root', async () => {
    inputs.localFilePath = ['resource/uploadDir/file1.txt'];
    inputs.obsFilePath = 'elif.txt'
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('elif.txt')).toBeGreaterThan(-1);
});

test('upload a nonexist file to obs root', async () => {
    inputs.localFilePath = ['file2.txt'];
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('file2.txt')).toEqual(-1);
});

test('upload a big file to obs', async () => {
    inputs.localFilePath = ['bigFile.zip'];
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('src/bigFile.zip')).toEqual(-1);
});

// ------------------------------folder---------------------------------

test('upload a exist empty folder to obs "obsTest2"', async () => {
    inputs.localFilePath = ['resource/uploadDir/folder2'];
    inputs.obsFilePath = 'obsTest2/'
    inputs.includeSelfFolder = true;
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('obsTest2/folder2/')).toBeGreaterThan(-1);
});

test('upload a exist folder to obs "obsTest2" and include local folder "uploadDir" itself', async () => {
    inputs.localFilePath = ['resource/uploadDir'];
    inputs.obsFilePath = 'obsTest2/';
    inputs.includeSelfFolder = true;
    const obs = getObsClient(inputs);

    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('obsTest2/uploadDir/folder1/')).toBeGreaterThan(-1);
});

test('upload a nonexist folder to obs root ', async () => {
    inputs.localFilePath = ['resource/uploadDir111'];
    inputs.obsFilePath = ''
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('uploadDir111/')).toEqual(-1);
});

test('upload a exist folder include lots of files to obs "obsTest3" ', async () => {
    inputs.localFilePath = ['resource/uploadDir/multi-files'];
    inputs.obsFilePath = 'obsTest3';
    const obs = getObsClient(inputs);
    await upload.uploadFileOrFolder(obs, inputs);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('obsTest3/multi-files/')).toEqual(-1);
});

// ----------------------------------------funciton----------------------------------------

test('fileDisplay', async () => {
    const uploadList = {
        file: [],
        folder: []
    };
    await upload.fileDisplay(getObsClient(inputs), inputs, 'resource/uploadDir/folder1', '', uploadList);
    expect(uploadList.file.length).toEqual(2);
    expect(uploadList.folder).toEqual(['folder1-1', 'folder1-1/folder1-1-1']);
});

test('getObsRootFile', () => {
    expect(upload.getObsRootFile(true, '/obs', 'local')).toEqual('obs/local');
    expect(upload.getObsRootFile(true, 'obs/', 'local')).toEqual('obs/local');
    expect(upload.getObsRootFile(true, 'obs', 'local')).toEqual('obs/local');
    expect(upload.getObsRootFile(true, '////obs///', 'local')).toEqual('obs/local');
    expect(upload.getObsRootFile(true, '', 'local')).toEqual('local');
    expect(upload.getObsRootFile(true, '/', 'local')).toEqual('local');

    expect(upload.getObsRootFile(false, '/obs', 'local')).toEqual('obs');
    expect(upload.getObsRootFile(false, 'obs/', 'local')).toEqual('obs');
    expect(upload.getObsRootFile(false, 'obs', 'local')).toEqual('obs');
    expect(upload.getObsRootFile(false, '////obs///', 'local')).toEqual('obs');
    expect(upload.getObsRootFile(false, '', 'local')).toEqual('');
    expect(upload.getObsRootFile(false, '/', 'local')).toEqual('');
});

test('obsCreateRootFolder', async () => {
    inputs.obsFilePath = 'obsTest2/newFolder';
    const obs = getObsClient(inputs);
    await upload.obsCreateRootFolder(obs, inputs.bucketName, inputs.obsFilePath);
    const objList = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(objList.indexOf('obsTest2/newFolder/') > -1).toBeTruthy();
});

test('formatObsPath', () => {
    expect(upload.formatObsPath('')).toEqual('');
    expect(upload.formatObsPath('/')).toEqual('');
    expect(upload.formatObsPath('/a/b')).toEqual('a/b');
    expect(upload.formatObsPath('/a/////\\b/c')).toEqual('a/b/c');
    expect(upload.formatObsPath('a/b//////c/')).toEqual('a/b/c/');
})