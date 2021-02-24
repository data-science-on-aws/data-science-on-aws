import json
import os
import time
import sys
from pip._internal import main

main(['install', '-I', '-q', 'boto3==1.16.47', '--target', '/tmp/', '--no-cache-dir', '--disable-pip-version-check'])
sys.path.insert(0,'/tmp/')

import boto3

region = boto3.Session().region_name
s3 = boto3.client('s3', region_name=region)
sm = boto3.client('sagemaker', region_name=region)

# Grab environment variables
PIPELINE_NAME = os.environ['PIPELINE_NAME']
print('Pipeline Name: {}'.format(PIPELINE_NAME))

timestamp = int(time.time())

def lambda_handler(event, context):
    print('boto3: {}'.format(boto3.__version__))
    print('Starting execution of pipeline {}...'.format(PIPELINE_NAME))
    
    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName='trigger-{}'.format(timestamp),
        PipelineParameters=[
        ],
        PipelineExecutionDescription= PIPELINE_NAME,
        # ClientRequestToken='string'
    )
    
    print('Response: {}'.format(response))
    
    execution_arn=response['PipelineExecutionArn']
    print('Pipeline Execution ARN: {}'.format(execution_arn))
    print('Done.')