"""Example workflow pipeline script for BERT pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

import os
import boto3
import logging
import time

from botocore.exceptions import ClientError

import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)

from sagemaker.workflow.step_collections import RegisterModel

sess   = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

sm = boto3.Session().client(service_name='sagemaker', region_name=region)

timestamp = str(int(time.time() * 10**7))

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print('BASE_DIR: {}'.format(BASE_DIR))


def get_pipeline(
    region,
    role=role,
    bucket=None,
    model_package_group_name="DSOAWS_BERT_PackageGroup",
    pipeline_name="DSOAWS_BERT_Pipeline",
    base_job_prefix="BERT",
):
    """Gets a SageMaker ML Pipeline instance working with BERT.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.c5.2xlarge"
    )
    train_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    train_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.c5.9xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://sagemaker-us-east-1-231218423789/amazon-reviews-pds/tsv/",
    )
    
    processor = SKLearnProcessor(
        framework_version='0.20.0',
        role=role,
        instance_type='ml.c5.2xlarge',
        instance_count=1,
        max_runtime_in_seconds=7200)

    # processing step for feature engineering
#     sklearn_processor = SKLearnProcessor(
#         framework_version="0.23-1",
#         instance_type=processing_instance_type,
#         instance_count=processing_instance_count,
#         base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
#         sagemaker_session=sagemaker_session,
#         role=role,
#     )

    ## DEFINE PROCESSING HYPERPARAMATERS  
    max_seq_length=64
    train_split_percentage=0.90
    validation_split_percentage=0.05
    test_split_percentage=0.05
    balance_dataset=True
    
    ## DEFINE PROCESSING INPUTS  
    raw_input_data_s3_uri = 's3://sagemaker-us-east-1-231218423789/amazon-reviews-pds/tsv/'
    print(raw_input_data_s3_uri)
    
    processing_inputs=[
        ProcessingInput(
            input_name='raw_input',
            source=raw_input_data_s3_uri,
            destination='/opt/ml/processing/input/data/',
            s3_data_distribution_type='ShardedByS3Key'
        )
    ]
    
    ## DEFINE PROCESSING OUTPUTS 
    processing_outputs=[
        ProcessingOutput(s3_upload_mode='EndOfJob',
                         output_name='bert-train',
                         source='/opt/ml/processing/output/bert/train',
#                         destination=processed_train_data_s3_uri
                        ),
        ProcessingOutput(s3_upload_mode='EndOfJob',
                         output_name='bert-validation',
                         source='/opt/ml/processing/output/bert/validation',
#                         destination=processed_validation_data_s3_uri
                        ),
        ProcessingOutput(s3_upload_mode='EndOfJob',
                         output_name='bert-test',
                         source='/opt/ml/processing/output/bert/test',
#                         destination=processed_test_data_s3_uri
                        ),
    ]
       
    
    step_process = ProcessingStep(
        name="PreprocessCustomerReviewsData",
        processor=processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        job_arguments=[
            '--train-split-percentage', str(train_split_percentage),
            '--validation-split-percentage', str(validation_split_percentage),
            '--test-split-percentage', str(test_split_percentage),
            '--max-seq-length', str(max_seq_length),
            '--balance-dataset', str(balance_dataset)],
        code=os.path.join(BASE_DIR, "preprocess-scikit-text-to-bert.py")
    )
    
    ## DEFINE TRAINING HYPERPARAMETERS
    epochs=1
    learning_rate=0.00001
    epsilon=0.00000001
    train_batch_size=128
    validation_batch_size=128
    test_batch_size=128
    train_steps_per_epoch=50
    validation_steps=50
    test_steps=50
    train_volume_size=1024
    use_xla=True
    use_amp=True
    freeze_bert_layer=False
    enable_sagemaker_debugger=False
    enable_checkpointing=False
    enable_tensorboard=False
    input_mode='File'
    run_validation=True
    run_test=False
    run_sample_predictions=False
    
    ## SETUP METRICS TO TRACK MODEL PERFORMANCE
    metrics_definitions = [
        {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'accuracy: ([0-9\\.]+)'},
        {'Name': 'validation:loss', 'Regex': 'val_loss: ([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'val_accuracy: ([0-9\\.]+)'}
    ]
    
    ## GET TRAINING IMAGE
    
    model_path = f"s3://{sess.default_bucket()}/{base_job_prefix}/TrainBERTModel"

    from sagemaker.tensorflow import TensorFlow

    image_uri = sagemaker.image_uris.retrieve(
        framework="tensorflow",
        region=region,
        version="2.1.0",
        py_version="py3",
        instance_type="ml.c5.xlarge",
        image_scope="training"
    )
    print(image_uri)
    
    train_code=os.path.join(BASE_DIR, "tf_bert_reviews.py")

        
    ## DEFINE TF ESTIMATOR
    estimator = TensorFlow(
        entry_point=train_code,
#        source_dir='src',
        role=role,
        output_path=model_path,
#        base_job_name=training_job_name,
        instance_count=1,
        instance_type='ml.c5.9xlarge',
        volume_size=train_volume_size,
        image_uri=image_uri,
#        py_version='py3',
#        framework_version='2.1.0',
        hyperparameters={
            'epochs': epochs,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'train_batch_size': train_batch_size,
            'validation_batch_size': validation_batch_size,
            'test_batch_size': test_batch_size,
            'train_steps_per_epoch': train_steps_per_epoch,
            'validation_steps': validation_steps,
            'test_steps': test_steps,
            'use_xla': use_xla,
            'use_amp': use_amp,
            'max_seq_length': max_seq_length,
            'freeze_bert_layer': freeze_bert_layer,
            'enable_sagemaker_debugger': enable_sagemaker_debugger,
            'enable_checkpointing': enable_checkpointing,
            'enable_tensorboard': enable_tensorboard,
            'run_validation': run_validation,
            'run_test': run_test,
            'run_sample_predictions': run_sample_predictions},
        input_mode=input_mode,
        metric_definitions=metrics_definitions,
#        max_run=7200 # max 2 hours * 60 minutes seconds per hour * 60 seconds per minute
    )
    
    ## TRAINING STEP
    step_train = TrainingStep(
        name="TrainBERTModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "bert-train"
                ].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "bert-validation"
                ].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "bert-test"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )        
        }
    )
    
    ## DEFINE EVALUATION STEP
#     script_eval = ScriptProcessor(
#         image_uri=image_uri,
#         command=["python3"],
#         instance_type=processing_instance_type,
#         instance_count=1,
#         base_job_name=f"{base_job_prefix}/script-bert-eval",
#         sagemaker_session=sagemaker_session,
#         role=role,
#     )
#     evaluation_report = PropertyFile(
#         name="AbaloneEvaluationReport",
#         output_name="evaluation",
#         path="evaluation.json",
#     )
#     step_eval = ProcessingStep(
#         name="EvaluateBERTModel",
#         processor=script_eval,
#         inputs=[
#             ProcessingInput(
#                 source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#                 destination="/opt/ml/processing/model",
#             ),
#             ProcessingInput(
#                 source=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "test"
#                 ].S3Output.S3Uri,
#                 destination="/opt/ml/processing/test",
#             ),
#         ],
#         outputs=[
#             ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
#         ],
#         code=os.path.join(BASE_DIR, "evaluate.py"),
#         property_files=[evaluation_report],
#     )

    ## REGISTER MODEL
    
    model_package_group_name = f"BERT-Reviews-{timestamp}"

    # NOTE: in the future, the model package group will be created automatically if it doesn't exist
    
    sm.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription="BERT-Reviews",
    )
    print(model_package_group_name)

#     model_metrics = ModelMetrics(
#         model_statistics=MetricsSource(
#             s3_uri="{}/evaluation.json".format(
#                 step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
#             ),
#             content_type="application/json"
#         )
#     )

    ## GET INFERENCE IMAGE 
    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="tensorflow",
        region=region,
        version="2.1.0",
        py_version="py3",
        instance_type="ml.m5.4xlarge",
        image_scope="inference"
    )
    print(inference_image_uri)

    step_register = RegisterModel(
        name="RegisterBERTModel",
        estimator=estimator,
        image_uri=inference_image_uri, # we have to specify, by default it's using training image
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.4xlarge"],
        transform_instances=["ml.c5.18xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status='PendingManualApproval',
    )
    
    ## EVALUATING MODEL -- CONDITION STEP
#     cond_lte = ConditionLessThanOrEqualTo(
#         left=JsonGet(
#             step=step_eval,
#             property_file=evaluation_report,
#             json_path="regression_metrics.mse.value"
#         ),
#         right=6.0,
#     )
#     step_cond = ConditionStep(
#         name="CheckMSEAbaloneEvaluation",
#         conditions=[cond_lte],
#         if_steps=[step_register],
#         else_steps=[],
#     )

    ## CREATE PIPELINE
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
#             processing_instance_type,
#             processing_instance_count,
#             training_instance_type,
#             model_approval_status,
#             input_data,
        ],
        steps=[step_process, step_train, step_register],
        sagemaker_session=sess
    )
    return pipeline