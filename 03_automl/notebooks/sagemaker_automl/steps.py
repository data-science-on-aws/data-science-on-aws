"""SageMaker AutoPilot Helpers.

This package contains helper classes and functions that are used in the candidates definition notebook.
"""
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sagemaker.sklearn import SKLearn, SKLearnModel

from sagemaker_automl.common import AutoMLLocalCandidateStep
from sagemaker_automl.common import get_algo_image_uri


class AutoMLCandidateAlgoStep:
    """Represents the Algorithm compute step of an AutoML local run. Currently supported `xgboost`, `linear-learner`
    and `mlp`.
    """

    def __init__(self, name, training_resource_config, region, repo_version, inference_repo_version,
                 candidate_specific_static_hyperparameters=None):

        self.algo_name = name
        self.training_resource_config = training_resource_config
        self.candidate_specific_static_hps = candidate_specific_static_hyperparameters \
            if candidate_specific_static_hyperparameters else {}
        self.region = region
        self.repo_version = repo_version
        self.algo_image_uri = get_algo_image_uri(self.algo_name, region, repo_version)
        self.algo_inference_image_uri = get_algo_image_uri(self.algo_name, region, inference_repo_version)

    def create_estimator(
        self, role, output_path, hyperparameters, sagemaker_session, **kwargs
    ):

        estimator = Estimator(
            self.algo_image_uri,
            role=role,
            instance_count=self.training_resource_config["instance_count"],
            instance_type=self.training_resource_config["instance_type"],
            output_path=output_path,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )
        hyperparameters.update(self.candidate_specific_static_hps)
        estimator.set_hyperparameters(**hyperparameters)

        return estimator

    def get_inference_container_config(self):
        config = {
            'env': {
                'SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT': 'text/csv'
            },
            'image_uri': self.algo_inference_image_uri
        }
        if self.algo_name == 'mlp':
            config['env']['ML_APPLICATION'] = 'mlp'
        elif self.algo_name == 'linear-learner':
            config['env']['ML_APPLICATION'] = 'linear_learner'

        return config


class AutoMLCandidateDataTransformerStep:
    """A DataTransformer step of a AutoML interative run candidate, representing the
    data processing pipeline(dpp) built with sagemaker scikit-learn automl container"""

    TRAIN_ENTRY_POINT = "trainer.py"
    SERVE_ENTRY_POINT = "candidate_data_processors/sagemaker_serve.py"
    DEFAULT_SOURCE_MODULE = "generated_module"

    # DEFATUL TRANSFORMER ENVIRONMENT. Please do not change the environment below.
    DEFAULT_TRANSFORMER_ENV = {
        "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv",
        "SAGEMAKER_PROGRAM": "sagemaker_serve",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
        "MAX_CONTENT_LENGTH": "20000000",
    }

    TRANSFORMED_DATA_FORMAT_SHORT_NAMES = {
        "text/csv": "csv",
        "application/x-recordio-protobuf": "recordio",
    }

    TRAIN_CHANNEL_NAME = "train"

    DEFAULT_TRANSFORMER_INPUT_MEDIA_TYPE = "text/csv"

    def __init__(
        self,
        name,
        training_resource_config,
        transform_resource_config,
        transforms_label,
        transformed_data_format,
        region,
        repo_version,
        source_module_path=None,
        sparse_encoding=False,
    ):
        self.name = name

        self.training_resource_config = training_resource_config
        self.transform_resource_config = transform_resource_config
        self.transforms_label = transforms_label
        self.transformed_data_format = transformed_data_format
        self.sparse_encoding = sparse_encoding

        self.source_module_path = source_module_path or self.DEFAULT_SOURCE_MODULE

        # We share registry account id with all framework container
        xgb_image_uri = image_uris.retrieve("xgboost", region=region, version="1.3-1")
        last_slash_index = xgb_image_uri.rfind('/')
        self.transformer_image_uri = "{}/{}:{}".format(
            xgb_image_uri[:last_slash_index], "sagemaker-sklearn-automl", repo_version
        )

    @property
    def train_instance_type(self):
        return self.training_resource_config["instance_type"]

    @property
    def train_instance_count(self):
        return self.training_resource_config["instance_count"]

    @property
    def train_volume_size_gb(self):
        return self.training_resource_config["volume_size_in_gb"]

    @property
    def transform_instance_type(self):
        return self.transform_resource_config["instance_type"]

    @property
    def transform_instance_count(self):
        return self.transform_resource_config["instance_count"]

    @property
    def content_type(self):
        return self.transformed_data_format

    @property
    def transformed_data_format_short(self):
        return self.TRANSFORMED_DATA_FORMAT_SHORT_NAMES[self.transformed_data_format]

    def create_trainer(
        self,
        output_path=None,
        role=None,
        hyperparameters=None,
        sagemaker_session=None,
        **kwargs,
    ):
        """Create a SKLearn trainer instance for our customized container
        Args:
            output_path (str): output path to store the trained model
            role (str): aws role arn
            hyperparameters: hyperparameters, currently empty
            sagemaker_session: an Sagemaker session object
            kwargs: other kwargs, not used.
        Returns:
            (SKLearn): a SKLearn trainer
        """
        _hyperparameters = hyperparameters or {}
        _hyperparameters.update({"processor_module": self.name})

        return SKLearn(
            entry_point=self.TRAIN_ENTRY_POINT,
            source_dir=f"{self.source_module_path}/candidate_data_processors",
            instance_type=self.train_instance_type,
            instance_count=self.train_instance_count,
            volume_size=self.train_volume_size_gb,
            image_uri=self.transformer_image_uri,
            output_path=output_path,
            hyperparameters=_hyperparameters,
            role=role,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )

    def create_steps(
        self,
        training_job_name,
        transform_job_name,
        transform_output_path,
        local_run_config,
        transformer_input_media_type=DEFAULT_TRANSFORMER_INPUT_MEDIA_TYPE,
    ):
        """This create a sequence of SageMaker jobs (e.g. training, batch transform) to be executed sequentially.

        Args:
            training_job_name (str): name of the training job name, used by trainer
            transform_job_name (str): name of the transform job name, used by batch transformer
            transform_output_path (str): output path of the transform job
            local_run_config (AutoMLLocalRunConfig): instance of AutoMLLocalRunConfig to provide some shared path
                convention, session etc
            transformer_input_media_type (str): default input type of transformers
        Return: a list of AutoMLLocalStep instances
        """

        def _train_transform(context):
            _trainer = context.get("trainer")

            training_data_input_path = (
                local_run_config.automl_job_processed_training_data_path
            )
            return _trainer.fit(
                {
                    AutoMLCandidateDataTransformerStep.TRAIN_CHANNEL_NAME: training_data_input_path
                },
                job_name=training_job_name,
                wait=True,
                logs=False,
            )

        # Create Transformer & Model
        def _create_transformer(context):
            _trainer = context.get("trainer")

            transform_env = dict(self.DEFAULT_TRANSFORMER_ENV)
            if self.sparse_encoding is True:
                transform_env["AUTOML_SPARSE_ENCODE_RECORDIO_PROTOBUF"] = "1"

            transform_env["SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT"] = self.content_type

            transformer = _trainer.transformer(
                instance_type=self.transform_instance_type,
                instance_count=self.transform_instance_count,
                output_path=transform_output_path,
                accept=self.content_type,
                env=transform_env,
                volume_kms_key=local_run_config.volume_kms_key,
                output_kms_key=local_run_config.output_kms_key
            )
            context["transformer"] = transformer

        def _transform_data(context):
            transformer = context.get("transformer")

            transformer.transform(
                local_run_config.automl_job_processed_data_path,
                job_name=transform_job_name,
                content_type=transformer_input_media_type,
                split_type="Line",
                wait=True,
                logs=False,
            )

        return [
            AutoMLLocalCandidateStep(
                name="train_data_transformer",
                action=_train_transform,
                description="SageMaker training job to learn the data transformations model",
            ),
            AutoMLLocalCandidateStep(
                name="create_transformer_model",
                action=_create_transformer,
                description="Create and save SageMaker model entity for the trained data transformer model",
            ),
            AutoMLLocalCandidateStep(
                name="perform_data_transform",
                action=_transform_data,
                description="Perform Batch transformation job to apply the trained "
                + "transformation model to the dataset to generate the algorithm compatible data",
            ),
        ]

    def create_model(
        self, estimator, role, sagemaker_session, transform_mode, **kwargs
    ):
        """Create a deployable data transformer model
        Args:
            estimator: an estimator attached from trainer
            sagemaker_session: sagemaker session
        Return: SKLearnModel object
        """

        environment = dict(self.DEFAULT_TRANSFORMER_ENV)
        environment["AUTOML_TRANSFORM_MODE"] = transform_mode or "feature-transform"

        return SKLearnModel(
            model_data=estimator.model_data,
            role=role,
            entry_point=f"{self.source_module_path}/{self.SERVE_ENTRY_POINT}",
            env=environment,
            image_uri=self.transformer_image_uri,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )
