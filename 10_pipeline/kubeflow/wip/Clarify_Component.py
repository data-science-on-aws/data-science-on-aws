#!/usr/bin/env python3
 
 
import kfp
import os
from kfp import components
from kfp import dsl
from kfp.components import create_component_from_func
from kfp.aws import use_aws_secret  
 
# Component function that checks for a particular value in SageMaker Clarify result 
def construct_analysis_config():
    import boto3
    import json
 
    sts = boto3.client("sts", region_name='eu-west-1')
    print(sts.get_caller_identity())
    s3 = boto3.client("s3", region_name='eu-west-1')
    
#     s3.create_bucket(Bucket='m-kfp-docs-trial-2', CreateBucketConfiguration={'LocationConstraint': "eu-west-1"})
 
 
 
construct_analysis_config_op = create_component_from_func(construct_analysis_config, base_image="python", packages_to_install=["boto3"])
 
 
@dsl.pipeline(
    name="SageMaker Clarify",
    description="SageMaker Clarify processing job",
)
def sagemaker_Clarify_analysis():
    region = "eu-west-1"
    analysis_config = construct_analysis_config_op(
    ).set_display_name("SageMaker Clarify Config").apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))
 
 
# if __name__ == "__main__":
#     kfp.compiler.Compiler().compile(sagemaker_Clarify_analysis, __file__ + ".zip")
    
kfp_client=kfp.Client()
namespace="kubeflow-user-example-com"
run_id = kfp_client.create_run_from_pipeline_func(sagemaker_Clarify_analysis, namespace=namespace, arguments={}).run_id
print("Run ID: ", run_id)
