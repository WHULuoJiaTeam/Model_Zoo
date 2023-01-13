import { expect, test } from '@jest/globals';
import * as fs from 'fs';
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
    operationType: 'download',
    region: 'cn-north-7',
    obsFilePath: '',
    localFilePath: [''],
    includeSelfFolder: false,
    exclude: ['test']
}

// --------------------file---------------------
test('download a exist file to local and rename it', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/file1.txt';
    inputs.localFilePath = ['resource/downloadDir/localfile1.txt'];
    const obs = getObsClient(inputs);
    
    await download.downloadFileOrFolder(obs, inputs);
    const isExist = fs.existsSync('resource/downloadDir/localfile1.txt');
    expect(isExist).toBeTruthy();
});

test('download a exist file but exclude it', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/folder1/file1-1.txt';
    inputs.localFilePath = ['resource/downloadDir/localfile2.txt'];
    inputs.exclude = ['obsTest2/uploadDir/folder1/file1-1.txt']
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs).then(() => {
        const isExist = fs.existsSync('resource/downloadDir/localfile2.txt');
        expect(isExist).toBeFalsy();
    });
});

test('download a nonexist file to local', async () => {
    inputs.obsFilePath = 'uploadDir/fileaaabbb.txt';
    inputs.localFilePath = ['resource/downloadDir/localfile3.txt'];
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs).then(() => {
        const isExist = fs.existsSync('resource/downloadDir/localfile3.txt');
        expect(isExist).toBeFalsy();
    });
});

test('download a file to local that local has same name folder', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/folder1/folder1-1/file1-1-1.txt';
    inputs.localFilePath = ['resource/downloadDir/file1-1-1.txt'];
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs).then(() => {
        const isExist = fs.existsSync('resource/downloadDir/file1-1-1.txt/file1-1-1.txt');
        expect(isExist).toBeTruthy();
    });
});

// ------------------------folder-------------------------

test('download a exist folder to local and exclude folder "folder1-1-1" and file "file1-1.txt"', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/';
    inputs.localFilePath = ['resource/downloadDir/local1/'];
    inputs.exclude = ['obsTest2/uploadDir/folder1/file1-1.txt', 'obsTest2/uploadDir/folder1/folder1-1/folder1-1-1/'];
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs).then(() => {
        const isExist1 = fs.existsSync('resource/downloadDir/local1/folder1/folder1-1/file1-1-1.txt');
        const isExist2 = fs.existsSync('resource/downloadDir/local1/folder1/file1-1.txt');
        expect(isExist1).toBeTruthy();
        expect(isExist2).toBeFalsy();
    });
});

test('download a exist folder to local and include folder itselt', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/';
    inputs.localFilePath = ['resource/downloadDir/local2/'];
    inputs.includeSelfFolder = true;
    inputs.exclude = ['tttttttttttttt'];
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs).then(() => {
        const isExist1 = fs.existsSync('resource/downloadDir/local2/uploadDir/file1.txt');
        const isExist2 = fs.existsSync('resource/downloadDir/local2/file1-1.txt');
        expect(isExist1).toBeTruthy();
        expect(isExist2).toBeFalsy();
    });
});

test('download a exist folder to nonexist local', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/folder1';
    inputs.localFilePath = ['resource/downloadDir/local3/'];
    inputs.includeSelfFolder = true;
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs);
    const isExist = fs.existsSync('resource/downloadDir/local3/folder1/');
    expect(isExist).toBeTruthy();
});

test('download a nonexist folder to local', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/folder321';
    inputs.localFilePath = ['resource/downloadDir/local3/'];
    inputs.includeSelfFolder = true;
    const obs = getObsClient(inputs);

    await download.downloadFileOrFolder(obs, inputs);
    const isExist = fs.existsSync('resource/downloadDir/local3/folder321/');
    expect(isExist).toBeFalsy();
});

test('download a folder to local that local has same name file', async () => {
    inputs.obsFilePath = 'obsTest2/uploadDir/folder1/';
    inputs.localFilePath = ['resource/downloadDir/local4/'];
    inputs.includeSelfFolder = true;
    const obs = getObsClient(inputs);

    await download.downloadFile(obs, inputs, inputs.obsFilePath, inputs.localFilePath[0]);
    expect(fs.lstatSync('resource/downloadDir/local4').isDirectory()).toBeFalsy();
});

// -----------------------------function--------------------------------

test('pathIsSingleFile', () => {
    expect(download.pathIsSingleFile(['obspath1.txt'], 'obspath1.txt')).toBeTruthy();
    expect(download.pathIsSingleFile(['obspath1.txt', 'obspath2'], 'obspath1.txt')).toBeFalsy();
    expect(download.pathIsSingleFile(['aaa/obspath1.txt'], 'aaa/obspath1.txt')).toBeTruthy();
    expect(download.pathIsSingleFile(['aaa/obspath1/'], 'aaa')).toBeFalsy();
});

test('getLocalFileName', () => {
    const local1 = download.getLocalFileName('resource/downloadDir/localDefault', 'obs1/obsfile.txt');
    expect(local1).toEqual('resource/downloadDir/localDefault/obsfile.txt');
    const local2 = download.getLocalFileName('resource/downloadDir/localDefault/1.txt', 'obs1/obsfile.txt');
    expect(local2).toEqual('resource/downloadDir/localDefault/1.txt');
});

test('getDownloadList and delUselessPath', async () => {
    inputs.obsFilePath = 'obsTest2';
    inputs.localFilePath = ['resource/downloadDir/local5'];
    inputs.includeSelfFolder = true;
    inputs.exclude = ['obsTest2/folder1-1', 'obsTest2/file1-1.txt']
    const obs = getObsClient(inputs);

    const res1 = await download.getDownloadList(obs, inputs, inputs.obsFilePath);
    expect(res1.indexOf('obsTest2/folder1-1/file1-1-1.txt')).toEqual(-1);
    expect(res1.indexOf('obsTest2/file1-1.txt')).toEqual(-1);
    expect(res1.indexOf('obsTest2/folder2/') > -1).toBeTruthy();
});
