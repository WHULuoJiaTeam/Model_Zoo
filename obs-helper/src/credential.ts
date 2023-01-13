import * as core from '@actions/core';

const HUAWEI_ClOUD_CREDENTIALS_ENVIRONMENT_VARIABLE_MAP = new Map<string, string>([
    ['access_key', 'HUAWEI_CLOUD_ACCESS_KEY_ID'],
    ['secret_key', 'HUAWEI_CLOUD_SECRET_ACCESS_KEY'],
    ['endpoint', 'HUAWEI_CLOUD_ENDPOINT'],
    ['region', 'HUAWEI_CLOUD_REGION'],
    ['project_id', 'HUAWEI_CLOUD_PROJECT_ID'],
]);

export function getCredential(param: string, isRequired: boolean): string {
    const environmentVariable = HUAWEI_ClOUD_CREDENTIALS_ENVIRONMENT_VARIABLE_MAP.get(param) || '';
    const credFromEnv = process.env[environmentVariable];
    const cred = credFromEnv ?? core.getInput(param, { required: false });
    if (isRequired && !cred) {
        core.setFailed(
            `The Huawei Cloud credential input ${param} is not correct. Please switch to using huaweicloud/auth-action which supports authenticating to Huawei Cloud.`
        );
    }
    return cred;
}
