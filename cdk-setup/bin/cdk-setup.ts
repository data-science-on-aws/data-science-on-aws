#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { CdkSetupStack } from '../lib/cdk-setup-stack';
import { Environment } from '@aws-cdk/core';

const app = new cdk.App();
const env: Environment = {
    account:'514975741450',
    region:'us-east-2'
}
new CdkSetupStack(app, 'CdkSetupStack', {env});
