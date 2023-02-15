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
