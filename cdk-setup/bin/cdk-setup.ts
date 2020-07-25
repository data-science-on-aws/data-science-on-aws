#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { CdkSetupStack } from '../lib/cdk-setup-stack';
import { Environment } from '@aws-cdk/core';

const app = new cdk.App();

const env: Environment = {
    account:'REPLACE_ME',
    region:'us-west-2',
}
new CdkSetupStack(app, 'CdkSetupStack', {env, createUser: true, explicitAccessPolicy:true, userName: 'REPLACE_ME'});
