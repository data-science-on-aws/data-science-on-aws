import boto3
import csv
import json
from time import gmtime, strftime

def lambda_handler(event, context):
    sagemaker = boto3.client('sagemaker')
    
    # TODO:  Change to not use the account number
    role = 'arn:aws:iam::835319576252:role/service-role/AmazonSageMaker-ExecutionRole-20191006T135881'
    bucket = 'sagemaker-us-east-1-835319576252'
    
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    processing_job_name = 'spark-preprocess-reviews-{}'.format(timestamp_prefix)
    print(processing_job_name)
    
    code_prefix = 'sagemaker/spark-preprocess-reviews-demo/code'
    py_files_prefix = 'sagemaker/spark-preprocess-reviews-demo/py_files'
    input_raw_prefix = 'sagemaker/spark-preprocess-reviews-demo/input/raw/reviews'
#    input_preprocessed_prefix = 'sagemaker/spark-preprocess-reviews-demo/{}/input/preprocessed/reviews'.format(timestamp_prefix)
    input_preprocessed_prefix = 'sagemaker/spark-preprocess-reviews-demo/input/preprocessed/reviews'

    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name

    ecr_repository = 'sagemaker-spark-example'
    tag = ':latest'
    spark_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account_id, region, ecr_repository + tag)
#    spark_repository_uri = '835319576252.dkr.ecr.us-east-1.amazonaws.com/sagemaker-spark-example:latest'
    print(spark_repository_uri)
    
    # Inputs
    s3_input_code = 's3://data-science-on-aws/{}/'.format(code_prefix)
    print(s3_input_code)

    s3_input_py_files = 's3://data-science-on-aws/{}/'.format(py_files_prefix)
    print(s3_input_py_files)
    
    s3_input_data = 's3://amazon-reviews-pds/parquet/'
    print(s3_input_data)
    
    # Outputs
    s3_output_train_data = 's3://{}/{}/train'.format(bucket, input_preprocessed_prefix)
    s3_output_validation_data = 's3://{}/{}/validation'.format(bucket, input_preprocessed_prefix)
    s3_output_test_data = 's3://{}/{}/test'.format(bucket, input_preprocessed_prefix)

    print(s3_output_train_data)
    print(s3_output_validation_data)
    print(s3_output_test_data)

    response = sagemaker.create_processing_job(
        ProcessingInputs=[
            {
                'InputName': 'reviews-code',
                'S3Input': {
                    'S3Uri': s3_input_code,
                    'LocalPath': '/opt/ml/processing/input/code/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            },
            {
                'InputName': 'reviews-py-files',
                'S3Input': {
                    'S3Uri': s3_input_py_files,
                    'LocalPath': '/opt/ml/processing/input/py_files/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                }
            },
        ],
        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'reviews-output-train',
                    'S3Output': {
                        'S3Uri': s3_output_train_data,
                        'LocalPath': '/opt/ml/processing/output/train',
                        'S3UploadMode': 'EndOfJob'
                    }
                },
                {
                    'OutputName': 'reviews-output-validation',
                    'S3Output': {
                        'S3Uri': s3_output_validation_data,
                        'LocalPath': '/opt/ml/processing/output/validation',
                        'S3UploadMode': 'EndOfJob'
                    }
                },
                {
                    'OutputName': 'reviews-output-test',
                    'S3Output': {
                        'S3Uri': s3_output_test_data,
                        'LocalPath': '/opt/ml/processing/output/test',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },
        ProcessingJobName=processing_job_name,
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 2,
                'InstanceType': 'ml.r5.xlarge',
                'VolumeSizeInGB': 10,
            }
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 600
        },
        AppSpecification={
            'ImageUri': spark_repository_uri,
            'ContainerEntrypoint': [
                '/opt/program/submit'
            ],
            'ContainerArguments': [
                '/opt/ml/processing/input/code/preprocess.py',
                's3_input_data', s3_input_data,
                's3_output_train_data', s3_output_train_data,
                's3_output_validation_data', s3_output_validation_data,
                's3_output_test_data', s3_output_test_data
            ],
        },
        Environment={
            'mode': 'python'
        },
        NetworkConfig={
            'EnableNetworkIsolation': False
        },
        RoleArn=role
    )

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }