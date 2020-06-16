"""SageMaker AutoPilot Helpers.

This package contains helper classes and functions that are used in the candidates definition notebook.
"""
import json
from os.path import join

import sagemaker


class AutoMLLocalRunConfig:
    """
    AutoMLLocalRunConfig represents common configurations and SageMaker constructs like role & sessions which are needed
    to start an AutoML local run job (e.g. from notebook).
    """

    # Those below are conventional outputs from AutoML job. Changes are only needed when there's breaking change of
    # storage path conventions from AutoML job.
    PRE_PROCESSED_DATA_ROOT = "preprocessed-data/tuning_data"
    PRE_PROCESSED_TRAINING_DATA_PATH = "train"
    PRE_PROCESSED_VALIDATION_DATA_PATH = "validation"

    def __init__(
        self,
        role,
        base_automl_job_config,
        local_automl_job_config,
        security_config=None,
        sagemaker_session=None,
    ):
        """Initialize an AutoMLLocalRunConfig

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            base_automl_job_config (dict): a dictionary that contains base AutoML job config which is generated from a
                managed automl run.
            local_automl_job_config (dict): a dictionary that contains inputs/outputs path convention of local run
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """

        self.role = role
        self.sagemaker_session = sagemaker_session or sagemaker.session.Session()
        self.region = self.sagemaker_session.boto_region_name

        self.base_automl_job_config = base_automl_job_config
        self.local_automl_job_config = local_automl_job_config

        # the job name of an existing AutoML managed run
        self.automl_job_name = base_automl_job_config["automl_job_name"]
        # the base s3 path where the managed AutoML job stores the intermediates (e.g. data transformation pipeline
        # candidate)
        self.automl_output_s3_base_path = base_automl_job_config[
            "automl_output_s3_base_path"
        ]

        # Auto ML output job path convention
        self.automl_job_processed_data_path = join(
            self.automl_output_s3_base_path, self.PRE_PROCESSED_DATA_ROOT
        )
        self.automl_job_processed_training_data_path = join(
            self.automl_job_processed_data_path, self.PRE_PROCESSED_TRAINING_DATA_PATH
        )
        self.automl_job_processed_validation_data_path = join(
            self.automl_job_processed_data_path, self.PRE_PROCESSED_VALIDATION_DATA_PATH
        )

        # Auto ML local job config
        self.local_automl_job_name = local_automl_job_config["local_automl_job_name"]
        self.local_automl_job_output_s3_base_path = local_automl_job_config[
            "local_automl_job_output_s3_base_path"
        ]

        # data transformer docker image repo version
        self.data_transformer_image_repo_version = base_automl_job_config[
            "data_transformer_image_repo_version"
        ]
        self.algo_image_repo_versions = base_automl_job_config[
            "algo_image_repo_versions"
        ]

        # The default conventional path to store the output from an local run job on S3.
        self.data_processing_model_s3_root = join(
            self.local_automl_job_output_s3_base_path,
            local_automl_job_config["data_processing_model_dir"],
        )
        self.transformed_output_s3_root = join(
            self.local_automl_job_output_s3_base_path,
            local_automl_job_config["data_processing_transformed_output_dir"],
        )
        self.tuning_output_s3_root = join(
            self.local_automl_job_output_s3_base_path,
            local_automl_job_config["multi_algo_tuning_output_dir"],
        )

        # Security config, note we invoke AutoML Boto API to get those configurations
        self.security_config = security_config or {}

    @property
    def vpc_config(self):
        return self.security_config.get("VpcConfig", None)

    @property
    def subnets(self):
        return (
            self.vpc_config.get("Subnets", None)
            if self.vpc_config is not None
            else None
        )

    @property
    def security_group_ids(self):
        return (
            self.vpc_config.get("SecurityGroupIds", None)
            if self.vpc_config is not None
            else None
        )

    @property
    def encrypt_inter_container_traffic(self):
        return self.security_config.get("EnableInterContainerTrafficEncryption", False)

    @property
    def volume_kms_key(self):
        return self.security_config.get("VolumeKmsKeyId", None)

    @property
    def output_kms_key(self):
        return self.security_config.get("OutputKmsKeyId", None)

    def to_dict(self):
        """
        Returns:
            dict: a dictionary representation of the instance
        """
        return {
            "role": self.role,
            "local_run_input": dict(
                **self.base_automl_job_config,
                train=self.automl_job_processed_training_data_path,
                validation=self.automl_job_processed_validation_data_path
            ),
            "local_run_output": dict(
                local_automl_job_name=self.local_automl_job_name,
                data_processing_model=self.data_processing_model_s3_root,
                transformed_output=self.transformed_output_s3_root,
                multi_algo_tuning_output=self.tuning_output_s3_root,
            ),
            "security_config": self.security_config,
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_html_table(self):
        return """
        <table>
        <tr><th colspan=2>Name</th><th>Value</th></tr>
        <tr><th>General</th><th>Role</th><td>{role}</td></tr>
        <tr><th rowspan=2>Base AutoML Job</th><th>Job Name</th><td>{base_job_name}</td></tr>
        <tr><th>Base Output S3 Path</th><td>{base_output_path}</td></tr>
        <tr><th rowspan=5>Interactive Job</th><th>Job Name</th><td>{local_job_name}</td></tr>
        <tr><th>Base Output S3 Path</th><td>{local_job_base_path}</td></tr>
        <tr><th>Data Processing Trained Model Directory</th><td>{dpp_model_dir}</td></tr>
        <tr><th>Data Processing Transformed Output</th><td>{dpp_transformed_data_dir}</td></tr>
        <tr><th>Algo Tuning Model Output Directory</th><td>{algo_tuning_output_model_dir}</td></tr>
        </table>
        """.format(
            role=self.role,
            base_job_name=self.automl_job_name,
            base_output_path=self.automl_output_s3_base_path,
            local_job_name=self.local_automl_job_name,
            local_job_base_path=self.local_automl_job_output_s3_base_path,
            dpp_model_dir=self.data_processing_model_s3_root,
            dpp_transformed_data_dir=self.transformed_output_s3_root,
            algo_tuning_output_model_dir=self.tuning_output_s3_root,
        )

    def display(self):
        from IPython.display import display, Markdown

        display(
            Markdown(
                "This notebook is initialized to use the following configuration: "
                + self.to_html_table()
            )
        )
