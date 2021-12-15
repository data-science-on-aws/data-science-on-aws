# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import time
import boto3
import logging
import cfnresponse
from botocore.exceptions import ClientError

sm_client = boto3.client('sagemaker')
logger = logging.getLogger(__name__)

def delete_user_profile(config):    
    domain_id = config['DomainId']
    user_profile_name = config['UserProfileName']
    logging.info(f'Start deleting user profile: {user_profile_name}')

    try:
        sm_client.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
    except:
        logging.info(f'Cannot retrieve {user_profile_name}')
        return

    for p in sm_client.get_paginator('list_apps').paginate(DomainIdEquals=domain_id, UserProfileNameEquals=user_profile_name):
        for a in p['Apps']:
            if a['Status'] != 'Deleted':
                sm_client.delete_app(DomainId=a['DomainId'], UserProfileName=a['UserProfileName'], AppType=a['AppType'], AppName=a['AppName'])
        
    apps = 1
    while apps:
        apps = 0
        for p in sm_client.get_paginator('list_apps').paginate(DomainIdEquals=domain_id, UserProfileNameEquals=user_profile_name):
            apps += len([a['AppName'] for a in p['Apps'] if a['Status'] != 'Deleted'])
        logging.info(f'Number of active apps: {str(apps)}')
        time.sleep(5)

    sm_client.delete_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)

    try:
        while sm_client.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name):
            time.sleep(5)
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFound':
            logger.info(f'{user_profile_name} deleted')
            return

def handler(event, context):
    response_data = {}
    physicalResourceId = event.get('PhysicalResourceId')
    config = event.get('ResourceProperties')

    try:
        if event['RequestType'] in ['Create', 'Update']:
            f = sm_client.update_user_profile
            if event['RequestType'] == 'Create':
                f = sm_client.create_user_profile

            response = f(DomainId=config['DomainId'], UserProfileName=config['UserProfileName'], UserSettings=config['UserSettings'])

            while response.get('Status') != 'InService':
                response = sm_client.describe_user_profile(DomainId=config['DomainId'], UserProfileName=config['UserProfileName'])
                time.sleep(5)
            
            response_data = {'UserProfileName':response['UserProfileName']}
            physicalResourceId = response['UserProfileName']
   
        elif event['RequestType'] == 'Delete':        
            delete_user_profile(config)

        cfnresponse.send(event, context, cfnresponse.SUCCESS, response_data, physicalResourceId=physicalResourceId)

    except ClientError as exception:
        logging.error(exception)
        cfnresponse.send(event, context, cfnresponse.FAILED, response_data, physicalResourceId=physicalResourceId, reason=str(exception))