import * as core from '@actions/core';
import * as fs from 'fs';
import { BucketInputs, CommonInputs, ObjectInputs } from './types';

/**
 * 目前支持obs功能的region列表
 * LA-Santiago	la-south-2
 * 非洲-约翰内斯堡	af-south-1
 * 华北-北京四	cn-north-4
 * 华北-北京一	cn-north-1
 * 华东-上海二	cn-east-2
 * 华东-上海一	cn-east-3
 * 华南-广州	cn-south-1
 * 拉美-墨西哥城二	la-north-2
 * 拉美-墨西哥城一	na-mexico-1
 * 拉美-圣保罗一	sa-brazil-1
 * 亚太-曼谷	ap-southeast-2
 * 亚太-新加坡	ap-southeast-3
 * 中国-香港	ap-southeast-1
 */
const regionArray = [
    'la-south-2',
    'af-south-1',
    'cn-north-4',
    'cn-north-1',
    'cn-east-2',
    'cn-east-3',
    'cn-south-1',
    'la-north-2',
    'na-mexico-1',
    'sa-brazil-1',
    'ap-southeast-2',
    'ap-southeast-3',
    'ap-southeast-1',
    'cn-central-221',
];

/**
 * 目前支持的存储类型
 * 标准存储 StorageClassStandard
 * 低频访问存储 StorageClassWarm
 * 归档存储 StorageClassCold
 */
const storageClassList = {
    standard: 'StorageClassStandard',
    infrequent: 'StorageClassWarm',
    archive: 'StorageClassCold',
};

/**
 * 目前支持的操作类型
 * 对象操作 upload  download
 * 桶操作 createbucket  deletebucket
 */
const OPERATION_TYPE = {
    object: ['upload', 'download'],
    bucket: ['createbucket', 'deletebucket'],
};

/**
 * 允许上传的最大文件大小（单位：B）
 */
const FILE_MAX_SIZE = 5 * 1024 * 1024 * 1024;

/**
 * 分段上传的段大小（单位：B）
 */
export const PART_MAX_SIZE = 1024 * 1024;

/**
 * 用于验证输入参数的正则表达式
 */
const akReg = /^[a-zA-Z0-9]{10,30}$/;
const skReg = /^[a-zA-Z0-9]{30,50}$/;
const legalReg = /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/;
const symbolReg = /([.]+[.-]+)|([-]+[.]+)/;
const ipReg = /(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}/;

/**
 * 检查ak/sk是否合法
 * @param ak
 * @param sk
 * @returns
 */
export function checkAkSk(ak: string, sk: string): boolean {
    return akReg.test(ak) && skReg.test(sk);
}

/**
 * 检查region是否合法
 * @param region
 * @returns
 */
export function checkRegion(region: string): boolean {
    return regionArray.includes(region);
}

/**
 * 检查桶名，规则如下
 * 3～63个字符，数字或字母开头，支持小写字母、数字、“-”、“.”
 * 禁止以“-”或“.”开头及结尾，禁止两个“.”相邻，禁止“.”和“-”相邻
 * 禁止类IP地址
 * @param bucketName
 * @returns
 */
export function checkBucketName(bucketName: string): boolean {
    return legalReg.test(bucketName) && !symbolReg.test(bucketName) && !ipReg.test(bucketName);
}

/**
 * 获得操作类型
 * @param operation_type
 * @returns
 */
export function getOperationCategory(operationType: string): string {
    if (OPERATION_TYPE.object.includes(operationType.toLowerCase())) {
        return 'object';
    }
    if (OPERATION_TYPE.bucket.includes(operationType.toLowerCase())) {
        return 'bucket';
    }
    return '';
}

/**
 * 检查上传时的input_file_path和参数obs_file_path是否合法
 * @param inputs
 * @returns
 */
export function checkUploadFilePath(inputs: ObjectInputs): boolean {
    if (inputs.localFilePath.length === 0) {
        core.setFailed('please input localFilePath.');
        return false;
    }
    if (inputs.localFilePath.length > 10) {
        core.setFailed('you should input no more than 10 local_file_path.');
        return false;
    }
    for (const path of inputs.localFilePath) {
        if (path === '') {
            core.setFailed('you should not input a empty string as local_file_path.');
            return false;
        }
        if (!fs.existsSync(path)) {
            core.setFailed(`local file or directory not exist, please check your input path.`);
            return false;
        }
    }
    return true;
}

/**
 * 检查下载时的input_file_path和obs_file_path是否合法
 * @param inputs
 * @returns
 */
export function checkDownloadFilePath(inputs: ObjectInputs): boolean {
    if (inputs.localFilePath.length !== 1) {
        core.setFailed('you should input one local_file_path.');
        return false;
    }
    if (inputs.localFilePath[0] === '') {
        core.setFailed('you should not input a empty string as local_file_path.');
        return false;
    }
    if (!inputs.obsFilePath) {
        core.setFailed('you should input one obs_file_path.');
        return false;
    }
    return true;
}

/**
 * 获得存储类型
 * @param storageClass
 * @returns
 */
export function getStorageClass(storageClass: string): string {
    for (const key in storageClassList) {
        if (storageClass === key) {
            return storageClassList[key as keyof typeof storageClassList];
        }
    }
    return '';
}

/**
 * 检查公共属性(ak,sk,region,bucketName)是否合法
 * @param inputs
 * @returns
 */
export function checkCommonInputs(inputs: CommonInputs): boolean {
    if (!checkAkSk(inputs.accessKey, inputs.secretKey)) {
        core.setFailed('ak or sk is not correct.');
        return false;
    }
    if (!checkRegion(inputs.region)) {
        core.setFailed('region is not correct.');
        return false;
    }
    if (!checkBucketName(inputs.bucketName)) {
        core.setFailed('bucket name is not correct.');
        return false;
    }
    return true;
}

/**
 * 检查操作对象时输入的参数(localFilePath,obsFilePath)是否合法
 * @param inputs
 * @returns
 */
export function checkObjectInputs(inputs: ObjectInputs): boolean {
    const checkFilePath =
        inputs.operationType.toLowerCase() === 'upload'
            ? checkUploadFilePath(inputs)
            : checkDownloadFilePath(inputs);
    if (!checkFilePath) {
        return false;
    }
    return true;
}

/**
 * 检查操作桶时输入的参数(storageClass)是否合法
 * @param inputs
 * @returns
 */
export function checkBucketInputs(inputs: BucketInputs): boolean {
    if (inputs.storageClass) {
        if (getStorageClass(inputs.storageClass) === '') {
            core.setFailed('storageClass is not correct.');
            return false;
        }
    }
    return true;
}

/**
 * 将传入的路径中的'\'替换为'/'
 * @param path
 * @returns
 */
export function replaceSlash(path: string): string {
    return path.replace(/\\/g, '/');
}

/**
 * 获得以rootPath开头， 并删除rootPath的path
 * 用于获得从obs下载对象时，对象应下载在本地的相对路径
 * @param rootPath
 * @param path
 * @returns
 */
export function getPathWithoutRootPath(rootPath: string, path: string): string {
    try {
        const aimPath = path.match(`^${rootPath}`);
        if (aimPath) {
            return path.replace(aimPath[0], '');
        } else {
            return path;
        }
    } catch (error) {
        core.info('rootPath not start with path.');
        return path;
    }
}

/**
 * 创建文件夹
 * @param path
 */
export function createFolder(path: string): boolean {
    if (fs.existsSync(path)) {
        return true;
    }
    try {
        fs.mkdirSync(path);
    } catch (error) {
        return false;
    }
    return fs.existsSync(path);
}

/**
 * 判断路径是否以'/'结尾
 * @param path
 * @returns
 */
export function isEndWithSlash(path: string): boolean {
    try {
        return path.slice(-1) === '/';
    } catch (error) {
        return false;
    }
}

/**
 * 删除字符串末尾的字符'/'
 * @param str
 * @returns
 */
export function getStringDelLastSlash(str: string): string {
    if (str) {
        return isEndWithSlash(str) ? str.substring(0, str.length - 1) : str;
    }
    return str;
}

/**
 * 检查文件是否超过5GB
 * @param filepath
 * @returns
 */
export function isFileOverSize(filepath: string): boolean {
    return fs.lstatSync(filepath).size > FILE_MAX_SIZE;
}

/**
 * 检查本地是否存在同名文件夹
 * @param localPath
 * @returns
 */
export function isExistSameNameFolder(localPath: string): boolean {
    return fs.existsSync(localPath) && fs.statSync(localPath).isDirectory();
}

/**
 * 检查本地是否存在同名文件
 *
 * @param localPath
 * @returns
 */
export function isExistSameNameFile(localPath: string): boolean {
    return (
        fs.existsSync(getStringDelLastSlash(localPath)) &&
        fs.statSync(getStringDelLastSlash(localPath)).isFile()
    );
}
