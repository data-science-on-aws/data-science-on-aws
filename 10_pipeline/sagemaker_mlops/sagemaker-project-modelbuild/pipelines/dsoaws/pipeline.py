"""
Example workflow pipeline script for BERT pipeline.

                                                 . -RegisterModel
                                                .
    Process-> Train -> (Evaluate -> Condition) .
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
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.properties import PropertyFile

from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput


sess = sagemaker.Session()
bucket = sess.default_bucket()
timestamp = int(time.time())

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print("BASE_DIR: {}".format(BASE_DIR))


def get_pipeline(region, role, default_bucket, pipeline_name, model_package_group_name, base_job_prefix):
    """Gets a SageMaker ML Pipeline instance working with BERT.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
        pipeline_name:  name of this pipeline
        model_package_group_name:  model package group
        base_job_prefix:  prefic of the job name

    Returns:
        an instance of a pipeline
    """

    sm = boto3.Session().client(service_name="sagemaker", region_name=region)

    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://{}/amazon-reviews-pds/tsv/".format(bucket),
    )

    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.c5.2xlarge")

    max_seq_length = ParameterInteger(
        name="MaxSeqLength",
        default_value=64,
    )

    balance_dataset = ParameterString(
        name="BalanceDataset",
        default_value="True",
    )

    train_split_percentage = ParameterFloat(
        name="TrainSplitPercentage",
        default_value=0.90,
    )

    validation_split_percentage = ParameterFloat(
        name="ValidationSplitPercentage",
        default_value=0.05,
    )

    test_split_percentage = ParameterFloat(
        name="TestSplitPercentage",
        default_value=0.05,
    )

    feature_store_offline_prefix = ParameterString(
        name="FeatureStoreOfflinePrefix",
        default_value="reviews-feature-store-" + str(timestamp),
    )

    feature_group_name = ParameterString(
        name="FeatureGroupName", default_value="reviews-feature-group-" + str(timestamp)
    )

    train_instance_type = ParameterString(name="TrainInstanceType", default_value="ml.c5.9xlarge")

    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=1)

    #########################
    # PROCESSING STEP
    #########################

    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        max_runtime_in_seconds=7200,
    )

    processing_inputs = [
        ProcessingInput(
            input_name="raw-input-data",
            source=input_data,
            destination="/opt/ml/processing/input/data/",
            s3_data_distribution_type="ShardedByS3Key",
        )
    ]

    processing_outputs = [
        ProcessingOutput(
            output_name="bert-train",
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/output/bert/train",
        ),
        ProcessingOutput(
            output_name="bert-validation",
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/output/bert/validation",
        ),
        ProcessingOutput(
            output_name="bert-test",
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/output/bert/test",
        ),
    ]

    # TODO:  Figure out why the Parameter's are not resolving properly to their native type when user here.
    #        We shouldn't be using `default_value`
    processing_step = ProcessingStep(
        name="Processing",
        processor=processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        job_arguments=[
            "--train-split-percentage",
            str(train_split_percentage.default_value),
            "--validation-split-percentage",
            str(validation_split_percentage.default_value),
            "--test-split-percentage",
            str(test_split_percentage.default_value),
            "--max-seq-length",
            str(max_seq_length.default_value),
            "--balance-dataset",
            str(balance_dataset.default_value),
            "--feature-store-offline-prefix",
            str(feature_store_offline_prefix.default_value),
            "--feature-group-name",
            str(feature_group_name.default_value),
        ],
        code=os.path.join(BASE_DIR, "preprocess-scikit-text-to-bert-feature-store.py"),
    )

    #########################
    # TRAINING STEP
    #########################

    epochs = ParameterInteger(name="Epochs", default_value=1)

    learning_rate = ParameterFloat(name="LearningRate", default_value=0.00001)

    epsilon = ParameterFloat(name="Epsilon", default_value=0.00000001)

    train_batch_size = ParameterInteger(name="TrainBatchSize", default_value=128)

    validation_batch_size = ParameterInteger(name="ValidationBatchSize", default_value=128)

    test_batch_size = ParameterInteger(name="TestBatchSize", default_value=128)

    train_steps_per_epoch = ParameterInteger(name="TrainStepsPerEpoch", default_value=50)

    validation_steps = ParameterInteger(name="ValidationSteps", default_value=50)

    test_steps = ParameterInteger(name="TestSteps", default_value=50)

    train_volume_size = ParameterInteger(name="TrainVolumeSize", default_value=1024)

    use_xla = ParameterString(
        name="UseXLA",
        default_value="True",
    )

    use_amp = ParameterString(
        name="UseAMP",
        default_value="True",
    )

    freeze_bert_layer = ParameterString(
        name="FreezeBERTLayer",
        default_value="False",
    )

    enable_sagemaker_debugger = ParameterString(
        name="EnableSageMakerDebugger",
        default_value="False",
    )

    enable_checkpointing = ParameterString(
        name="EnableCheckpointing",
        default_value="False",
    )

    enable_tensorboard = ParameterString(
        name="EnableTensorboard",
        default_value="False",
    )

    input_mode = ParameterString(
        name="InputMode",
        default_value="File",
    )

    run_validation = ParameterString(
        name="RunValidation",
        default_value="True",
    )

    run_test = ParameterString(
        name="RunTest",
        default_value="False",
    )

    run_sample_predictions = ParameterString(
        name="RunSamplePredictions",
        default_value="False",
    )

    metrics_definitions = [
        {"Name": "train:loss", "Regex": "loss: ([0-9\\.]+)"},
        {"Name": "train:accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        {"Name": "validation:loss", "Regex": "val_loss: ([0-9\\.]+)"},
        {"Name": "validation:accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
    ]

    train_src = os.path.join(BASE_DIR, "src")
    model_path = f"s3://{default_bucket}/{base_job_prefix}/output/model"

    estimator = TensorFlow(
        entry_point="tf_bert_reviews.py",
        source_dir=BASE_DIR,
        role=role,
        output_path=model_path,
        instance_count=train_instance_count,
        instance_type=train_instance_type,
        volume_size=train_volume_size,
        py_version="py37",
        framework_version="2.3.1",
        hyperparameters={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "train_batch_size": train_batch_size,
            "validation_batch_size": validation_batch_size,
            "test_batch_size": test_batch_size,
            "train_steps_per_epoch": train_steps_per_epoch,
            "validation_steps": validation_steps,
            "test_steps": test_steps,
            "use_xla": use_xla,
            "use_amp": use_amp,
            "max_seq_length": max_seq_length,
            "freeze_bert_layer": freeze_bert_layer,
            "enable_sagemaker_debugger": enable_sagemaker_debugger,
            "enable_checkpointing": enable_checkpointing,
            "enable_tensorboard": enable_tensorboard,
            "run_validation": run_validation,
            "run_test": run_test,
            "run_sample_predictions": run_sample_predictions,
        },
        input_mode=input_mode,
        metric_definitions=metrics_definitions,
        #        max_run=7200 # max 2 hours * 60 minutes seconds per hour * 60 seconds per minute
    )

    training_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["bert-train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["bert-validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["bert-test"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    #########################
    # EVALUATION STEP
    #########################

    evaluation_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        max_runtime_in_seconds=7200,
    )

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="metrics", path="evaluation.json")

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        code=os.path.join(BASE_DIR, "evaluate_model_metrics.py"),
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingInputs["raw-input-data"].S3Input.S3Uri,
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/metrics/"
            ),
        ],
        job_arguments=[
            "--max-seq-length",
            str(max_seq_length.default_value),
        ],
        property_files=[evaluation_report],  # these cause deserialization issues
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    #########################
    ## REGISTER TRAINED MODEL STEP
    #########################

    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    deploy_instance_type = ParameterString(name="DeployInstanceType", default_value="ml.m5.4xlarge")

    deploy_instance_count = ParameterInteger(name="DeployInstanceCount", default_value=1)

    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="tensorflow",
        region=region,
        version="2.3.1",
        py_version="py37",
        instance_type=deploy_instance_type,
        image_scope="inference",
    )
    print(inference_image_uri)

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        image_uri=inference_image_uri,  # we have to specify, by default it's using training image
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[
            deploy_instance_type
        ],  # The JSON spec must be within these instance types or we will see "Instance Type Not Allowed" Exception
        transform_instances=[deploy_instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
    )

    #########################
    ## CREATE MODEL FOR DEPLOYMENT STEP
    #########################

    model = Model(
        image_uri=inference_image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sess,
        role=role,
    )

    create_inputs = CreateModelInput(
        instance_type=deploy_instance_type,
    )

    create_step = CreateModelStep(
        name="CreateModel",
        model=model,
        inputs=create_inputs,
    )

    #########################
    ## CONDITION STEP:  EVALUATE THE MODEL
    #########################

    min_accuracy_value = ParameterFloat(name="MinAccuracyValue", default_value=0.01)

    minimum_accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=evaluation_step,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",
        ),
        right=min_accuracy_value,  # accuracy
    )

    minimum_accuracy_condition_step = ConditionStep(
        name="AccuracyCondition",
        conditions=[minimum_accuracy_condition],
        if_steps=[register_step, create_step],  # success, continue with model registration
        else_steps=[],  # fail, end the pipeline
    )

    #########################
    ## CREATE PIPELINE
    #########################

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            processing_instance_count,
            processing_instance_type,
            max_seq_length,
            balance_dataset,
            train_split_percentage,
            validation_split_percentage,
            test_split_percentage,
            feature_store_offline_prefix,
            feature_group_name,
            train_instance_type,
            train_instance_count,
            epochs,
            learning_rate,
            epsilon,
            train_batch_size,
            validation_batch_size,
            test_batch_size,
            train_steps_per_epoch,
            validation_steps,
            test_steps,
            train_volume_size,
            use_xla,
            use_amp,
            freeze_bert_layer,
            enable_sagemaker_debugger,
            enable_checkpointing,
            enable_tensorboard,
            input_mode,
            run_validation,
            run_test,
            run_sample_predictions,
            min_accuracy_value,
            model_approval_status,
            deploy_instance_type,
            deploy_instance_count,
        ],
        steps=[processing_step, training_step, evaluation_step, minimum_accuracy_condition_step],
        sagemaker_session=sess,
    )

    #########################
    ## RETURN PIPELINE
    #########################

    return pipeline
