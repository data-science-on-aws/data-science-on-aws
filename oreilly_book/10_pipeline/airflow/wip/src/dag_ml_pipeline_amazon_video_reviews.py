from __future__ import print_function
import json
import requests
from datetime import datetime

# airflow operators
import airflow
from airflow.models import DAG
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

# airflow sagemaker operators
from airflow.contrib.operators.sagemaker_training_operator \
    import SageMakerTrainingOperator
from airflow.contrib.operators.sagemaker_tuning_operator \
    import SageMakerTuningOperator
from airflow.contrib.operators.sagemaker_transform_operator \
    import SageMakerTransformOperator
from airflow.contrib.hooks.aws_hook import AwsHook

# sagemaker sdk
import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner

# airflow sagemaker configuration
from sagemaker.workflow.airflow import training_config
from sagemaker.workflow.airflow import tuning_config
from sagemaker.workflow.airflow import transform_config_from_estimator

# ml workflow specific
from pipeline import prepare, preprocess
import config as cfg

# =============================================================================
# functions
# =============================================================================


def is_hpo_enabled():
    """check if hyper-parameter optimization is enabled in the config
    """
    hpo_enabled = False
    if "job_level" in config and \
            "run_hyperparameter_opt" in config["job_level"]:
        run_hpo_config = config["job_level"]["run_hyperparameter_opt"]
        if run_hpo_config.lower() == "yes":
            hpo_enabled = True
    return hpo_enabled


def get_sagemaker_role_arn(role_name, region_name):
    iam = boto3.client('iam', region_name=region_name)
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]

# =============================================================================
# setting up training, tuning and transform configuration
# =============================================================================


# read config file
config = cfg.config

# set configuration for tasks
hook = AwsHook(aws_conn_id='airflow-sagemaker')
region = config["job_level"]["region_name"]
sess = hook.get_session(region_name=region)
role = get_sagemaker_role_arn(
    config["train_model"]["sagemaker_role"],
    sess.region_name)
container = get_image_uri(sess.region_name, 'factorization-machines')
hpo_enabled = is_hpo_enabled()

# create estimator
fm_estimator = Estimator(
    image_name=container,
    role=role,
    sagemaker_session=sagemaker.session.Session(sess),
    **config["train_model"]["estimator_config"]
)

# train_config specifies SageMaker training configuration
train_config = training_config(
    estimator=fm_estimator,
    inputs=config["train_model"]["inputs"])

# create tuner
fm_tuner = HyperparameterTuner(
    estimator=fm_estimator,
    **config["tune_model"]["tuner_config"]
)

# create tuning config
tuner_config = tuning_config(
    tuner=fm_tuner,
    inputs=config["tune_model"]["inputs"])

# create transform config
transform_config = transform_config_from_estimator(
    estimator=fm_estimator,
    task_id="model_tuning" if hpo_enabled else "model_training",
    task_type="tuning" if hpo_enabled else "training",
    **config["batch_transform"]["transform_config"]
)

# =============================================================================
# define airflow DAG and tasks
# =============================================================================

# define airflow DAG

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    dag_id='sagemaker-ml-pipeline',
    default_args=args,
    schedule_interval=None,
    concurrency=1,
    max_active_runs=1,
    user_defined_filters={'tojson': lambda s: json.JSONEncoder().encode(s)}
)

# set the tasks in the DAG

# dummy operator
init = DummyOperator(
    task_id='start',
    dag=dag
)

# preprocess the data
preprocess_task = PythonOperator(
    task_id='preprocessing',
    dag=dag,
    provide_context=False,
    python_callable=preprocess.preprocess,
    op_kwargs=config["preprocess_data"])

# prepare the data for training
prepare_task = PythonOperator(
    task_id='preparing',
    dag=dag,
    provide_context=False,
    python_callable=prepare.prepare,
    op_kwargs=config["prepare_data"]
)

branching = BranchPythonOperator(
    task_id='branching',
    dag=dag,
    python_callable=lambda: "model_tuning" if hpo_enabled else "model_training")

# launch sagemaker training job and wait until it completes
train_model_task = SageMakerTrainingOperator(
    task_id='model_training',
    dag=dag,
    config=train_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30
)

# launch sagemaker hyperparameter job and wait until it completes
tune_model_task = SageMakerTuningOperator(
    task_id='model_tuning',
    dag=dag,
    config=tuner_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30
)

# launch sagemaker batch transform job and wait until it completes
batch_transform_task = SageMakerTransformOperator(
    task_id='predicting',
    dag=dag,
    config=transform_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30,
    trigger_rule=TriggerRule.ONE_SUCCESS
)

cleanup_task = DummyOperator(
    task_id='cleaning_up',
    dag=dag)

# set the dependencies between tasks

init.set_downstream(preprocess_task)
preprocess_task.set_downstream(prepare_task)
prepare_task.set_downstream(branching)
branching.set_downstream(tune_model_task)
branching.set_downstream(train_model_task)
tune_model_task.set_downstream(batch_transform_task)
train_model_task.set_downstream(batch_transform_task)
batch_transform_task.set_downstream(cleanup_task)
