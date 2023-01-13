import * as fs from 'fs';
import * as path from 'path';
import * as utils from '../utils';
import * as core from '@actions/core';
import { ObjectInputs, UploadFileList, SUCCESS_STATUS_CODE } from '../types';

/**
 * 上传文件/文件夹
 * @param obsClient  Obs客户端，因obsClient为引入的obs库的类型，本身并未导出其类型，故使用any，下同
 * @param inputs 用户输入的参数
 */
export async function uploadFileOrFolder(obsClient: any, inputs: ObjectInputs): Promise<void> {
    for (const localPath of inputs.localFilePath) {
        const localFilePath = utils.getStringDelLastSlash(localPath); // 去除本地路径参数结尾的'/'，方便后续使用
        const localName = path.basename(localFilePath);
        try {
            const fsStat = fs.lstatSync(localFilePath);
            if (fsStat.isFile()) {
                await uploadFile(obsClient, inputs.bucketName, localFilePath, getObsFilePath(inputs, localName));
            }

            if (fsStat.isDirectory()) {
                const obsFileRootPath = getObsRootFile(!!inputs.includeSelfFolder, inputs.obsFilePath, localName);
                const uploadList = {
                    file: [],
                    folder: [],
                };
                await fileDisplay(obsClient, inputs, localFilePath, obsFileRootPath, uploadList);

                // 若总文件数大于1000，取消上传
                const uploadListLength = uploadList.file.length + uploadList.folder.length;
                if (uploadListLength > 1000) {
                    core.setFailed(`local directory: '${localPath}' has ${uploadListLength} files and folders,`);
                    core.setFailed(`please upload a directory include less than 1000 files and folders.`);
                    return;
                }

                if (inputs.obsFilePath) {
                    await obsCreateRootFolder(
                        obsClient,
                        inputs.bucketName,
                        utils.getStringDelLastSlash(inputs.obsFilePath)
                    );
                }
                await uploadFileAndFolder(obsClient, inputs.bucketName, uploadList);
            }
        } catch (error) {
            core.setFailed(`read local file or directory: '${localPath}' failed.`);
        }
    }
}

/**
 * 得到待上传对象在obs的根路径
 * @param includeSelf 是否包含文件夹自身
 * @param obsfile 用户输入的obs_file_path
 * @param objectName 对象在本地的名称
 * @returns
 */
export function getObsRootFile(includeSelf: boolean, obsfile: string, objectName: string): string {
    const formatPath = formatObsPath(obsfile);
    if (includeSelf) {
        return formatPath ? `${utils.getStringDelLastSlash(formatPath)}/${objectName}` : objectName;
    } else {
        return utils.getStringDelLastSlash(formatPath);
    }
}

/**
 * 格式化上传时的obs_file_path
 * @param obsPath 
 * @returns 
 */
export function formatObsPath(obsPath: string): string {
    // 上传到根目录
    if (!obsPath || utils.replaceSlash(path.normalize(obsPath)) === '/') {
        return '';
    }
    const pathFormat = utils.replaceSlash(path.normalize(obsPath));
    return pathFormat.startsWith('/') ? pathFormat.substring(1) : pathFormat;
}

/**
 * 得到待上传文件在obs的路径
 * @param inputs 用户输入的参数
 * @param localRoot 待上传文件的根目录
 * @returns
 */
export function getObsFilePath(inputs: ObjectInputs, localRoot: string): string {
    if (!inputs.obsFilePath) {
        return localRoot;
    }

    if (utils.isEndWithSlash(inputs.obsFilePath)) {
        return `${inputs.obsFilePath}${localRoot}`;
    } else {
        // 若是多路径上传时的文件,不存在重命名,默认传至obs_file_path文件夹下
        return inputs.localFilePath.length > 1 ? `${inputs.obsFilePath}/${localRoot}` : inputs.obsFilePath;
    }
}

/**
 * 读取文件夹, 统计待上传文件/文件夹路径
 * @param obsClient Obs客户端
 * @param inputs 用户输入的参数
 * @param localFilePath 本地路径
 * @param obsFileRootPath 要上传到obs的根路径
 * @param uploadList 待上传文件列表
 */
export async function fileDisplay(
    obsClient: any,
    inputs: ObjectInputs,
    localFilePath: string,
    obsFileRootPath: string,
    uploadList: UploadFileList
): Promise<void> {
    const fslist = fs.readdirSync(localFilePath);
    if (fslist.length > 0) {
        for (const filename of fslist) {
            // 得到当前文件的绝对路径
            const filepath = path.join(localFilePath, filename);
            const info = fs.statSync(filepath);
            const obsFilePath = obsFileRootPath ? `${obsFileRootPath}/${filename}` : `${obsFileRootPath}${filename}`;

            if (info.isFile()) {
                uploadList.file.push({
                    local: utils.replaceSlash(path.normalize(filepath)),
                    obs: obsFilePath,
                });
            }
            
            if (info.isDirectory()) {
                uploadList.folder.push(obsFilePath);
                await fileDisplay(obsClient, inputs, filepath, obsFilePath, uploadList);
            }
        }
    } else {
        // 是空文件夹
        if (uploadList.folder.indexOf(utils.getStringDelLastSlash(obsFileRootPath)) === -1) {
            uploadList.folder.push(utils.getStringDelLastSlash(obsFileRootPath));
        }
    }
}

/**
 * 上传文件和文件夹
 * @param obsClient Obs客户端
 * @param bucketName 桶名
 * @param uploadList 待上传对象列表
 */
async function uploadFileAndFolder(obsClient: any, bucketName: string, uploadList: UploadFileList): Promise<void> {
    for (const folder of uploadList.folder) {
        await uploadFolder(obsClient, bucketName, folder);
    }
    for (const file of uploadList.file) {
        await uploadFile(obsClient, bucketName, file.local, file.obs);
    }
}

/**
 * 上传文件
 * @param obsClient Obs客户端
 * @param bucketName 桶名
 * @param localFilePath 对象在本地的路径
 * @param obsFilePath 对象要上传到obs的路径
 * @returns
 */
export async function uploadFile(
    obsClient: any,
    bucketName: string,
    localFilePath: string,
    obsFilePath: string
): Promise<void> {
    if (utils.isFileOverSize(localFilePath)) {
        core.setFailed(`your local file "${localFilePath}" cannot be uploaded because it is larger than 5 GB`);
        return;
    }
    core.info(`start upload file: "${localFilePath}"`);
    const result = await obsClient.putObject({
        Bucket: bucketName,
        Key: obsFilePath,
        SourceFile: localFilePath,
    });
    if (result.CommonMsg.Status < SUCCESS_STATUS_CODE) {
        core.info(`succeessfully upload file: "${localFilePath}"`);
    } else {
        core.setFailed(`failed to upload file: "${localFilePath}", because ${result.CommonMsg.Code}`);
    }
}

/**
 * 上传文件夹
 * 因obs无实际文件夹概念, 不需要本地路径, 只需目标路径即可
 * @param obsClient Obs客户端
 * @param bucketName 桶名
 * @param obsFilePath 对象要上传到obs的路径
 * @returns
 */
export async function uploadFolder(obsClient: any, bucketName: string, obsFilePath: string): Promise<void> {
    core.info(`start create folder "${obsFilePath}/"`);
    const result = await obsClient.putObject({
        Bucket: bucketName,
        Key: `${obsFilePath}/`,
    });
    if (result.CommonMsg.Status < SUCCESS_STATUS_CODE) {
        core.info(`succeessfully create folder "${obsFilePath}/"`);
    } else {
        core.setFailed(`failed to create folder "${obsFilePath}/", because ${result.CommonMsg.Code}`);
    }
}

/**
 * 在obs创建空文件夹
 * 上传时若指定一个obs上非已存在的路径, 则需要在obs上逐级建立文件夹
 * @param obsClient Obs客户端
 * @param bucketName 桶名
 * @param obsFilePath 对象要上传到obs的路径
 * @returns
 */
export async function obsCreateRootFolder(obsClient: any, bucketName: string, obsFile: string): Promise<void> {
    const obsPathList = obsFile.split('/');
    let obsPath = '';
    for (const path of obsPathList) {
        if (!path) {
            return;
        }
        obsPath += `${path}/`;
        core.info(`start create folder "${obsPath}"`);
        const result = await obsClient.putObject({
            Bucket: bucketName,
            Key: obsPath,
        });
        if (result.CommonMsg.Status < SUCCESS_STATUS_CODE) {
            core.info(`succeessfully create folder "${obsPath}"`);
        } else {
            core.setFailed(`failed to create folder "${obsPath}", because ${result.CommonMsg.Code}`);
        }
    }
}

/**
 * 分段上传
 * @param obs obs客户端
 * @param bucketName 桶名
 * @param objKey 上传对象在obs上的名称
 * @param filePath 上传对象的本地路径
 */
export async function multipartUpload(obs: any, bucketName: string, objKey: string, filePath: string): Promise<void> {
    const uploadId = await initMultipartUpload(obs, bucketName, objKey);
    if (uploadId) {
        const parts = await uploadParts(obs, bucketName, objKey, uploadId, filePath);
        if (parts.length > 0) {
            await mergeParts(obs, bucketName, objKey, uploadId, parts);
        }
    }
}

/**
 * 初始化分段上传任务
 * @param obs obs客户端
 * @param bucketName 桶名
 * @param objKey 上传对象在obs上的名称
 * @returns
 */
export async function initMultipartUpload(obs: any, bucketName: string, objKey: string): Promise<string> {
    const result = await obs.initiateMultipartUpload({
        Bucket: bucketName,
        Key: objKey,
    });

    if (result.CommonMsg.Status < SUCCESS_STATUS_CODE) {
        core.info('init multipart upload successfully.');
        return result.InterfaceResult.UploadId;
    } else {
        core.setFailed('init multipart upload failed.');
        return '';
    }
}

/**
 * 上传分段
 * @param obs obs客户端
 * @param bucketName 桶名
 * @param objKey 上传对象在obs上的名称
 * @param uploadId 分段上传任务的uploadid
 * @param filePath 上传对象的本地路径
 * @returns
 */
export async function uploadParts(
    obs: any,
    bucketName: string,
    objKey: string,
    uploadId: string,
    filePath: string
): Promise<{ PartNumber: number; ETag: any }[]> {
    const partSize = utils.PART_MAX_SIZE;

    const fileLength = fs.lstatSync(filePath).size;
    const partCount =
        fileLength % partSize === 0 ? Math.floor(fileLength / partSize) : Math.floor(fileLength / partSize) + 1;

    core.info(`total parts count ${partCount}.`);

    const parts: { PartNumber: number; ETag: any }[] = [];

    core.info('Begin to upload multiparts to OBS from a file');
    for (let i = 0; i < partCount; i++) {
        const offset = i * partSize;
        const currPartSize = i + 1 === partCount ? fileLength - offset : partSize;
        const partNumber = i + 1;

        const result = await obs.uploadPart({
            Bucket: bucketName,
            Key: objKey,
            PartNumber: partNumber,
            UploadId: uploadId,
            SourceFile: filePath,
            Offset: offset,
            PartSize: currPartSize,
        });
        if (result.CommonMsg.Status < SUCCESS_STATUS_CODE) {
            parts.push({
                PartNumber: partNumber,
                ETag: result.InterfaceResult.ETag,
            });
        } else {
            core.setFailed(result.CommonMsg.Code);
        }
    }

    if (parts.length === partCount) {
        // Sort parts order by partNumber asc
        const _parts = parts.sort((a, b) => {
            if (a.PartNumber >= b.PartNumber) {
                return 1;
            }
            return -1;
        });
        return _parts;
    }
    return parts;
}

/**
 * 合并分段
 * @param obs obs客户端
 * @param bucketName 桶名
 * @param objKey 上传对象在obs上的名称
 * @param uploadId 分段上传任务的uploadid
 * @param parts 分段上传任务的分段信息
 * @returns
 */
export async function mergeParts(
    obs: any,
    bucketName: string,
    objKey: string,
    uploadId: string,
    parts: any[]
): Promise<boolean> {
    const result = await obs.completeMultipartUpload({
        Bucket: bucketName,
        Key: objKey,
        UploadId: uploadId,
        Parts: parts,
    });

    if (result.CommonMsg.Status < SUCCESS_STATUS_CODE) {
        core.info('Complete to upload multiparts finished.');
        return true;
    } else {
        core.setFailed(result.CommonMsg.Code);
        return false;
    }
}
