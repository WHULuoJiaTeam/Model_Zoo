import * as fs from 'fs';
import path from 'path';
import * as core from '@actions/core';
import * as bucket from './bucket';
import * as utils from '../utils';
import { ObjectInputs, ListBucketContentItem } from '../types';

/**
 * 下载文件或者文件夹
 * @param obsClient Obs客户端，因obsClient为引入的obs库的类型，本身并未导出其类型，故使用any，下同
 * @param inputs 用户输入的参数
 * @returns
 */
export async function downloadFileOrFolder(obsClient: any, inputs: ObjectInputs): Promise<void> {
    const inputLocalFilePath = utils.replaceSlash(path.normalize(inputs.localFilePath[0]));
    const downloadPathList = await getDownloadList(obsClient, inputs, inputs.obsFilePath);

    if (downloadPathList.length < 1) {
        core.setFailed('object not exist in obs or no object needed downloaded.');
        return;
    }

    if (pathIsSingleFile(downloadPathList, inputs.obsFilePath)) {
        // 下载单个文件时，以'/'结尾，代表下载到localpath代表的文件夹中，下载文件夹时无此限制
        const fileLocalPath = utils.isEndWithSlash(inputLocalFilePath)
            ? `${inputLocalFilePath}${path.basename(downloadPathList[0])}`
            : inputLocalFilePath;
        await downloadFile(obsClient, inputs, downloadPathList[0], fileLocalPath);
    } else {
        await downloadFilesFromObs(obsClient, inputs, downloadPathList, inputLocalFilePath);
    }
}

/**
 * 待下载的对象是否为单个文件
 * @param downloadPathList 待下载的对象列表
 * @param obsPath 对象在obs上的path
 * @returns
 */
export function pathIsSingleFile(downloadPathList: string[], obsPath: string): boolean {
    return (
        downloadPathList.length === 1 && downloadPathList[0] === obsPath && !utils.isEndWithSlash(downloadPathList[0])
    );
}

/**
 * 下载多个文件/文件夹
 * @param obsClient Obs客户端
 * @param inputs 用户输入的参数
 * @param downloadList 待下载列表
 * @param localPath 本地path
 */
async function downloadFilesFromObs(
    obsClient: any,
    inputs: ObjectInputs,
    downloadList: string[],
    localPath: string
): Promise<void> {
    const localRoot = getDownloadRoot(localPath, inputs.obsFilePath, !!inputs.includeSelfFolder);
    
    if (localRoot !== '' && utils.replaceSlash(path.normalize(localRoot)) !== '/') {
        createLocalRootFolder(utils.replaceSlash(path.normalize(localRoot)));
    }

    let delFolderPath = ''; // 用来记录无法下载的文件夹
    for (const path of downloadList) {
        if (delFolderPath === '' || !path.match(`^${delFolderPath}`)) {
            let finalLocalPath = `${localRoot}${utils.getPathWithoutRootPath(
                utils.getStringDelLastSlash(inputs.obsFilePath),
                path
            )}`;

            // 若本地有和待下载文件同名的文件夹，给文件名加后缀下载
            if (downloadList.indexOf(`${finalLocalPath}/`) !== -1) {
                finalLocalPath = `${finalLocalPath}${new Date().valueOf()}`;
            }

            // 下载文件/文件夹
            if (utils.isEndWithSlash(finalLocalPath)) {
                // 若本地有和待下载文件夹同名的文件，停止下载此文件夹
                if (utils.isExistSameNameFile(finalLocalPath)) {
                    core.info(
                        `download folder "${finalLocalPath}" failed, because there is already a file with the same name".`
                    );
                    delFolderPath = path;
                } else {
                    delFolderPath = downloadFolder(finalLocalPath);
                }
            } else {
                await downloadFile(obsClient, inputs, path, finalLocalPath);
            }
        }
    }
}

/**
 * 根据includeSelfFolder，获取本地文件根目录
 * @param localPath 本地path
 * @param obsPath 对象在obs上的path
 * @param includeSelfFolder 是否包含文件夹自身
 * @returns
 */
export function getDownloadRoot(localPath: string, obsPath: string, includeSelfFolder: boolean): string {
    return includeSelfFolder
        ? `${utils.getStringDelLastSlash(localPath)}/${utils.getStringDelLastSlash(obsPath).split('/').pop()}`
        : utils.getStringDelLastSlash(localPath);
}

/**
 * 下载文件夹时，检查并创建本地根目录
 * @param localPath
 */
export function createLocalRootFolder(localPath: string): void {
    let local = localPath.startsWith('/') ? '/' : '';

    for (const dir of localPath.split('/')) {
        local = path.join(local, dir)
        utils.createFolder(local);
    }
}

/**
 * 下载文件夹
 * @param localPath 文件夹要下载在本地的路径
 * @returns
 */
export function downloadFolder(localPath: string): string {
    const isCreated = utils.createFolder(localPath);
    if (isCreated) {
        return '';
    } else {
        core.setFailed(`failed to create folder: "${localPath}"`);
        return localPath;
    }
}

/**
 * 下载文件
 * @param obsClient Obs客户端
 * @param inputs 用户输入的参数
 * @param obsPath 对象在obs上的路径
 * @param localPath 文件要下载在本地的路径
 */
export async function downloadFile(
    obsClient: any,
    inputs: ObjectInputs,
    obsPath: string,
    localPath?: string
): Promise<void> {
    let localFileName = localPath ?? getLocalFileName(utils.getStringDelLastSlash(inputs.localFilePath[0]), obsPath);

    // 若本地存在同名文件夹，下载到此文件夹中。若此文件夹中还存在同名文件夹，放弃本次下载
    if (utils.isExistSameNameFolder(localFileName)) {
        core.info(
            `a folder already exists on the local that has the same name as the path for downloading the obs file "${obsPath}"`
        );
        core.info(`try to download obs file in this folder`);
        const nextFileName = `${localFileName}/${path.basename(localFileName)}`;
        if (utils.isExistSameNameFolder(nextFileName)) {
            core.info(
                `download file "${localFileName}" failed, because "${localFileName}" already exists as a folder on the local.`
            );
            return;
        } else {
            localFileName = nextFileName;
        }
    }

    core.info(`start download obs file: "${obsPath}"`);
    const result = await obsClient.getObject({
        Bucket: inputs.bucketName,
        Key: obsPath,
        SaveAsFile: localFileName,
    });
    if (result.CommonMsg.Status < 300) {
        core.info(`successfully download obs file: "${obsPath}"`);
    } else {
        core.setFailed(`failed to download obs file: "${obsPath}", because ${result.CommonMsg.Code}`);
    }
}

/**
 * 获得对象应在本地的路径
 * @param localPath 本地文件夹路径
 * @param obsPath 对象在obs上的路径
 * @returns
 */
export function getLocalFileName(localPath: string, obsPath: string): string {
    try {
        if (fs.lstatSync(localPath).isDirectory()) {
            return `${localPath}/${path.basename(obsPath)}`;
        } else {
            return localPath;
        }
    } catch (error) {
        return localPath;
    }
}

/**
 * 获取在obs上待下载的对象列表
 * 官方提供的getObject方法最大请求1000个文件，若请求的文件大于1000个则返回对象名按照字典序排序后的前1000个文件
 * result.InterfaceResult.IsTruncated表明本次请求是否返回了全部结果，“true”表示没有返回全部结果；“false”表示已返回了全部结果
 * result.InterfaceResult.NextMarker会记录下次起始位置
 * @param obsClient Obs客户端
 * @param inputs 用户输入的参数
 * @param obsPath 对象在obs上的路径
 * @returns
 */
export async function getDownloadList(obsClient: any, inputs: ObjectInputs, obsPath: string): Promise<string[]> {
    const obsFilePath = utils.getStringDelLastSlash(obsPath);

    let resultList: string[] = [];
    let isTruncated = true;
    let marker = '';

    while (isTruncated) {
        const result = await bucket.listObjects(obsClient, inputs.bucketName, obsFilePath, marker);
        resultList = resultList.concat(delUselessPath(result.InterfaceResult.Contents, inputs));

        isTruncated = result.InterfaceResult.IsTruncated === 'true';
        marker = result.InterfaceResult.NextMarker;
    }
    return resultList;
}

/**
 * 从待下载列表中排除不需要的对象（用户输入的排除项）
 * @param objList 列举出的桶内对象列表
 * @param inputs 用户输入的参数
 * @returns
 */
function delUselessPath(objList: ListBucketContentItem[], inputs: ObjectInputs): string[] {
    const resultList: string[] = [];
    objList.forEach((element: ListBucketContentItem) => {
        // 删除不需要的path，仅保留inputs.obsFilePath相关的文件路径
        let isInclude = true;
        if (!!inputs.exclude && inputs.exclude.length > 0) {
            inputs.exclude.forEach((excludeItem) => {
                if (excludeItem && element['Key'].search(`^${utils.getStringDelLastSlash(excludeItem)}`) > -1) {
                    isInclude = false;
                }
            });
        }
        if (isInclude) {
            resultList.push(element['Key']);
        }
    });
    return resultList;
}
