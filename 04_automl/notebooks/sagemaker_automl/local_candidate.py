"""SageMaker AutoPilot Helpers.

This package contains helper classes and functions that are used in the candidates definition notebook.
"""
import os
import json

from sagemaker_automl.common import (
    AutoMLLocalCandidateNotPrepared,
    AutoMLLocalCandidateNotTrained,
    uid,
)
from sagemaker_automl.steps import (
    AutoMLCandidateAlgoStep,
    AutoMLCandidateDataTransformerStep,
)


class AutoMLLocalCandidate:
    """AutoMLLocalCandidate models an AutoML pipeline consist of data transformer and algo steps
    """

    def __init__(
        self, candidate_name, data_transformer_step, algo_step, local_run_config
    ):
        """
        Args:
            candidate_name (str): name of the candidate, e.g. `dpp0-xgboost`
            data_transformer_step: the data transformer step of the candidate
            algo_step: the algo step of the candidate
            local_run_config (AutoMLLocalRunConfig): an instance of AutoMLLocalRunConfig
        """
        self.candidate_name = candidate_name
        self.algo_name = algo_step.algo_name

        self.data_transformer_step = data_transformer_step
        self.algo_step = algo_step

        self.local_run_config = local_run_config

        self._state = {}

    @classmethod
    def create(cls, candidate_name, candidate_definition, local_run_config):
        """Factory method to create an AutoMLLocalCandidate from a candidate_definition

        Args:
            candidate_name (str): name of the candidate AutoML pipeline
            candidate_definition (dict): a dictionary representing the candidate_definition
                ```
                {
                    "data_transformer": {
                        "name": "dpp0",
                        "training_resource_config": {
                            "instance_type": "ml.m5.2xlarge",
                            "instance_count": 1,
                            "volume_size_in_gb": 50
                        },
                        "transform_resource_config": {
                            "instance_type": "ml.m5.2xlarge",
                            "instance_count": 1
                        },
                        "transforms_label": True,
                        "transformed_data_format": 'text/csv',
                        "sparse_encoding": False
                    },
                    "algorithm": {
                        "name": "xgboost",
                        "training_resource_config": {
                            "instance_type": "ml.m5.2xlarge",
                            "instance_count": 1
                        }
                    }
                }
                ```
            local_run_config (AutoMLLocalRunConfig): an AutoMLLocalRunConfig
        :return: an instance of AutoMLLocalCandidate
        """

        data_transformer_step = AutoMLCandidateDataTransformerStep(
            **candidate_definition["data_transformer"],
            region=local_run_config.region,
            repo_version=local_run_config.data_transformer_image_repo_version,
            source_module_path=os.path.join(
                f"{local_run_config.automl_job_name}-artifacts",
                AutoMLCandidateDataTransformerStep.DEFAULT_SOURCE_MODULE)
        )

        algo_name = candidate_definition["algorithm"]["name"]
        algo_step = AutoMLCandidateAlgoStep(
            **candidate_definition["algorithm"],
            region=local_run_config.region,
            repo_version=local_run_config.algo_image_repo_versions[algo_name]
        )

        return AutoMLLocalCandidate(
            candidate_name, data_transformer_step, algo_step, local_run_config
        )

    @property
    def content_type(self):
        return self.data_transformer_step.content_type

    @property
    def transforms_label(self):
        return self.data_transformer_step.transforms_label

    @property
    def data_transformer_transformed_data_path(self):
        self._check_data_transformer_prepared()
        return self._state["data_transformer"]["transform_output_path"]

    def prepare_data_transformers_for_training(
        self, training_job_name=None, transform_job_name=None, **kwargs
    ):
        """This prepare the data transformers for training:
        1. create SKlearn trainer
        2. create steps to be executed by runner

        Args:
            training_job_name (str):
                when specified we'll respect a training job name specified by user, otherwise we'll generate one
            transform_job_name (str):
                when specified we'll respect a batch transform job name specified by user, otherwise we'll generate one
            kwargs: everything `sagemaker.sklearn.SKLearn` accepts
        """

        # add network & security features
        kwargs[
            "encrypt_inter_container_traffic"
        ] = self.local_run_config.encrypt_inter_container_traffic

        kwargs["subnets"] = self.local_run_config.subnets
        kwargs["security_group_ids"] = self.local_run_config.security_group_ids
        kwargs["train_volume_kms_key"] = self.local_run_config.volume_kms_key
        kwargs["output_kms_key"] = self.local_run_config.output_kms_key

        data_transformer_trainer = self.data_transformer_step.create_trainer(
            output_path=self.local_run_config.data_processing_model_s3_root,
            role=self.local_run_config.role,
            sagemaker_session=self.local_run_config.sagemaker_session,
            **kwargs
        )

        training_job_name = (
            training_job_name
            or "{prefix}-{dpp_name}-train-{suffix}".format(
                prefix=self.local_run_config.local_automl_job_name,
                dpp_name=self.data_transformer_step.name,
                suffix=uid(),
            )
        )

        transform_job_name = (
            transform_job_name
            or "{prefix}-{dpp_name}-transform-{suffix}".format(
                prefix=self.local_run_config.local_automl_job_name,
                dpp_name=self.data_transformer_step.name,
                suffix=uid(),
            )
        )

        transform_output_path = "{prefix}/{dpp_name}/{transformed_data_format}".format(
            prefix=self.local_run_config.transformed_output_s3_root,
            dpp_name=self.data_transformer_step.name,
            transformed_data_format=self.data_transformer_step.transformed_data_format_short,
        )

        automl_steps = self.data_transformer_step.create_steps(
            training_job_name=training_job_name,
            transform_job_name=transform_job_name,
            local_run_config=self.local_run_config,
            transform_output_path=transform_output_path,
        )

        self._state["data_transformer"] = {
            "training_job_name": training_job_name,
            "transform_job_name": transform_job_name,
            "transform_output_path": transform_output_path,
            "trainer": data_transformer_trainer,
            "steps": automl_steps,
            "trained": False,
        }

    @property
    def data_transformer_steps(self):

        self._check_data_transformer_prepared()

        return self._state["data_transformer"]["steps"]

    def get_data_transformer_trainer(self):

        self._check_data_transformer_prepared()

        return self._state["data_transformer"]["trainer"]

    def _check_data_transformer_prepared(self):
        if "data_transformer" not in self._state:
            raise AutoMLLocalCandidateNotPrepared(
                "AutoML Candidate has not been initialized yet. please invoke "
                + "`prepare_data_transformers_for_training`() before getting the model"
            )

    def set_transformer_trained(self):
        self._state["data_transformer"]["trained"] = True

    def data_transformer_is_trained(self):
        return (
            "data_transformer" in self._state
            and self._state["data_transformer"]["trained"]
        )

    def get_data_transformer_model(
        self, role, sagemaker_session, transform_mode=None, **kwargs
    ):
        """

        Args:
            role: IAM role arn used to invoke API
            sagemaker_session: an SageMaker.session.Session() object
            transform_mode: transform mode of the data transformers
            kwargs: other parameters accepted by SKLearnModel

        Returns:
            (SKLearnModel): a trained data transformer model
        """

        self._check_data_transformer_prepared()

        if not self.data_transformer_is_trained:
            raise AutoMLLocalCandidateNotTrained(
                "AutoML Candidate data transformers has not been trained yet"
            )

        data_transformer_state = self._state["data_transformer"]

        trainer = data_transformer_state["trainer"]
        training_job_name = data_transformer_state["training_job_name"]

        data_transformer_estimator = trainer.attach(
            training_job_name, sagemaker_session=sagemaker_session
        )

        security_config = self.local_run_config.security_config

        if (
            self.local_run_config.security_config is not None
            and "VpcConfig" not in kwargs
        ):
            kwargs.update({"vpc_config": security_config["VpcConfig"]})

        return self.data_transformer_step.create_model(
            estimator=data_transformer_estimator,
            role=role,
            sagemaker_session=sagemaker_session,
            transform_mode=transform_mode,
            **kwargs
        )

    def to_dict(self):
        base_dict = {
            "pipeline_name": self.candidate_name,
            "data_transformer": {
                "data_processing_module_name": self.data_transformer_step.name
            },
            "algorithm": {"algo_name": self.algo_step.algo_name},
        }

        if "data_transformer" in self._state:
            base_dict["data_transformer"].update(
                {
                    "training_job_name": self._state["data_transformer"][
                        "training_job_name"
                    ],
                    "transform_job_name": self._state["data_transformer"][
                        "transform_job_name"
                    ],
                }
            )

        return base_dict

    def __repr__(self):

        return json.dumps(self.to_dict(), indent=4)
