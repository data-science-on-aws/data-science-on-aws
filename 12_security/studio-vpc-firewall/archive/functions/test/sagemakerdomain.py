# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import time
import boto3
import logging
import cfnresponse
from botocore.exceptions import ClientError

sm_client = boto3.client('sagemaker')
logger = logging.getLogger(__name__)

SAGEMAKER_DOMAIN_AUTH_MODE = 'IAM'
SAGEMAKER_NETWORK_ACCESS_TYPE = 'VpcOnly'
SAGEMAKER_EFS_RETENTION_POLICY = 'Delete'

def delete_domain(domain_id):
    try:
        sm_client.describe_domain(DomainId=domain_id)
    except:
        return

    sm_client.delete_domain(DomainId=domain_id,RetentionPolicy={'HomeEfsFileSystem': SAGEMAKER_EFS_RETENTION_POLICY})

    try:
        while sm_client.describe_domain(DomainId=domain_id):
            time.sleep(5)
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFound':
            logger.info(f'SageMaker domain {domain_id} has been deleted')
            return

def handler(event, context):
    response_data = {}
    physicalResourceId = event.get('PhysicalResourceId')
    config = event.get('ResourceProperties')

    try:
        if event['RequestType'] in ['Create', 'Update']:
            if event['RequestType'] == 'Create':
                physicalResourceId = sm_client.create_domain(
                                DomainName=config['DomainName'],
                                AuthMode=SAGEMAKER_DOMAIN_AUTH_MODE,
                                DefaultUserSettings=config['DefaultUserSettings'],
                                SubnetIds=config['SageMakerStudioSubnetIds'].split(','),
                                VpcId=config['VpcId'],
                                AppNetworkAccessType=SAGEMAKER_NETWORK_ACCESS_TYPE
                            )['DomainArn'].split('/')[-1]

                logger.info(f'Created SageMaker Studio Domain:{physicalResourceId}')
            else:
                sm_client.update_domain(DomainId=physicalResourceId, DefaultUserSettings=config['DefaultUserSettings'])

            while sm_client.describe_domain(DomainId=physicalResourceId)['Status'] != 'InService':
                time.sleep(5)

            response_data = {'DomainId': physicalResourceId}

        elif event['RequestType'] == 'Delete':
            delete_domain(physicalResourceId)

        cfnresponse.send(event, context, cfnresponse.SUCCESS, response_data, physicalResourceId=physicalResourceId)

    except ClientError as exception:
        logging.error(exception)
        cfnresponse.send(event, context, cfnresponse.FAILED, response_data, physicalResourceId=physicalResourceId, reason=str(exception))