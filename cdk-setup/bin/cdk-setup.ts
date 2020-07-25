#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { CdkSetupStack } from '../lib/cdk-setup-stack';

const app = new cdk.App();
new CdkSetupStack(app, 'CdkSetupStack');
