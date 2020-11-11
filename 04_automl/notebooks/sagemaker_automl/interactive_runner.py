"""SageMaker AutoPilot Helpers.

This package contains helper classes and functions that are used in the candidates definition notebook.
"""
import concurrent
import logging
import sys
from concurrent.futures.thread import ThreadPoolExecutor

import sagemaker

from sagemaker_automl.common import execute_steps
from sagemaker_automl.local_candidate import AutoMLLocalCandidate

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.getLevelName("INFO"),
)


class AutoMLInteractiveRunner:
    """AutoMLInteractiveRunner is an orchestrator that manages the AutoML local run. This includes the following:

        1. Manages the state of local candidates selection
        2. Orchestrate multi-algo tuning operations that requires inputs from all candidates.
        3. Model selection and export of trained estimator to deployable model
    """

    def __init__(self, local_run_config, candidates=None):
        """
        Args:
            local_run_config (AutoMLLocalRunConfig): an AutoMLLocalRunConfig instance
            candidates (dict): optional. Default to an empty dict
        """
        self.candidates = candidates or {}
        self.local_run_config = local_run_config

    def select_candidate(self, candidate_definition):
        """
        Args:
            candidate_definition (dict): Candidate definition in JSON
            ```
            {
                "data_transformer": {
                    "name": "dpp0",
                    "training_resource_config": {
                        "instance_type": "ml.m5.2xlarge",
                        "instance_count": 1
                    },
                    "transform_resource_config": {
                        "instance_type": "ml.m5.2xlarge",
                        "instance_count": 1
                    },
                    "transforms_label": True
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
        """

        # pipeline_name is derived from data_transformer and algorithm name

        candidate_pipeline_name = "{data_transformer_name}-{algo_name}".format(
            data_transformer_name=candidate_definition["data_transformer"]["name"],
            algo_name=candidate_definition["algorithm"]["name"],
        )

        if candidate_pipeline_name in self.candidates:
            logging.info(
                "Warning: pipeline candidate {} has already been selected, replacing".format(
                    candidate_pipeline_name
                )
            )

        # create candidate
        self.candidates[candidate_pipeline_name] = AutoMLLocalCandidate.create(
            candidate_name=candidate_pipeline_name,
            candidate_definition=candidate_definition,
            local_run_config=self.local_run_config,
        )

    def fit_data_transformers(self, parallel_jobs=2, start_jitter_seconds=10):
        """Fit data transformers from all candidates in parallel
        Args:
            parallel_jobs (int): num of parallel jobs to run
            start_jitter_seconds: jitter on executor start up to avoid throttling too fast when
                too many job are started
        """

        execution_future = {}

        with ThreadPoolExecutor(
            max_workers=parallel_jobs, thread_name_prefix="Worker"
        ) as executor:
            for candidate_pipeline_name, candidate in self.candidates.items():
                candidate.prepare_data_transformers_for_training()

                trainer = candidate.get_data_transformer_trainer()
                steps = candidate.data_transformer_steps

                # The worker future is the key and value is the candidate pipeline name
                # This is used to report progress below
                execution_future[
                    executor.submit(
                        execute_steps,
                        execution_name=candidate_pipeline_name,
                        steps=steps,
                        context={"trainer": trainer},
                        start_jitter_seconds=start_jitter_seconds,
                    )
                ] = candidate_pipeline_name

            iterator = concurrent.futures.as_completed(execution_future)

            success_count = 0

            try:
                while True:
                    future = next(iterator)
                    candidate_pipeline_name = execution_future[future]
                    success = self._process_data_transformer_future(
                        candidate_pipeline_name, future
                    )

                    if success:
                        success_count += 1

            except StopIteration:
                logging.info(
                    "Successfully fit {} data transformers".format(success_count)
                )

    def _process_data_transformer_future(self, candidate_pipeline_name, future):

        try:
            future.result()
            logging.info(
                "Successfully fit data transformer for {}".format(
                    candidate_pipeline_name
                )
            )
            self.candidates[candidate_pipeline_name].set_transformer_trained()
            return True
        except Exception:
            logging.error(
                "Failed to fit data transformer for {}".format(candidate_pipeline_name),
                exc_info=True,
            )

        return False

    def prepare_multi_algo_parameters(
        self,
        objective_metrics,
        static_hyperparameters,
        hyperparameters_search_ranges,
        **estimator_kwargs,
    ):
        """Prepare input parameters of multi algo tuning

        Args:
            objective_metrics (dict[str, str]): Name of the metric for evaluating training jobs.
            static_hyperparameters (dict[str, dict]): Static hyperparameters for the algorithm
            hyperparameters_search_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of
                parameter ranges. These parameter ranges can be one
                of three types: Continuous, Integer, or Categorical. The keys of
                the dictionary are the names of the hyperparameter, and the
                values are the appropriate parameter range class to represent
                the range.
            estimator_kwargs: other supported kwargs of `sagemaker.estimator import Estimator`
        """
        # Create Estimators

        estimator_kwargs[
            "encrypt_inter_container_traffic"
        ] = self.local_run_config.encrypt_inter_container_traffic

        estimator_kwargs["subnets"] = self.local_run_config.subnets
        estimator_kwargs[
            "security_group_ids"
        ] = self.local_run_config.security_group_ids
        estimator_kwargs["output_kms_key"] = self.local_run_config.output_kms_key
        estimator_kwargs["enable_network_isolation"] = True

        estimators = {
            candidate_name: candidate.algo_step.create_estimator(
                role=self.local_run_config.role,
                output_path="{s3_root}/{candidate_name}".format(
                    s3_root=self.local_run_config.tuning_output_s3_root,
                    candidate_name=candidate_name,
                ),
                hyperparameters=static_hyperparameters[candidate.algo_name],
                sagemaker_session=self.local_run_config.sagemaker_session,
                **estimator_kwargs,
            )
            for candidate_name, candidate in self.candidates.items()
            if candidate.data_transformer_is_trained()
        }

        # Objective_metrics
        objective_metrics_dict = {
            candidate_name: objective_metrics[candidate.algo_name]
            for candidate_name, candidate in self.candidates.items()
            if candidate.data_transformer_is_trained()
        }

        # HPO Hyperparameter ranges
        hyperparameter_search_ranges_dict = {
            candidate_name: hyperparameters_search_ranges[candidate.algo_name]
            for candidate_name, candidate in self.candidates.items()
            if candidate.data_transformer_is_trained()
        }

        return {
            "estimator_dict": estimators,
            "objective_metric_name_dict": objective_metrics_dict,
            "hyperparameter_ranges_dict": hyperparameter_search_ranges_dict,
        }

    def prepare_multi_algo_inputs(self):
        return {
            candidate_name: {
                "train": sagemaker.session.s3_input(
                    candidate.data_transformer_transformed_data_path + "/train",
                    content_type=candidate.content_type,
                ),
                "validation": sagemaker.session.s3_input(
                    candidate.data_transformer_transformed_data_path + "/validation",
                    content_type=candidate.content_type,
                ),
            }
            for candidate_name, candidate in self.candidates.items()
            if candidate.data_transformer_is_trained()
        }

    def choose_candidate(self, tuner_analytics_dataframe, multi_algo_training_job_name):
        """This choose a candidate from tuner analytics data frame based on candidate_training_job_name

        Args:
            tuner_analytics_dataframe: a dataframe from tuner analytics
            multi_algo_training_job_name: selected multi-algo training job name
        Returns: an AutoMLLocalCandidate
        """

        training_job_analytics = tuner_analytics_dataframe.loc[
            tuner_analytics_dataframe["TrainingJobName"] == multi_algo_training_job_name
        ]
        # The TrainingJobDefinitionName is mapped to candidate name
        best_data_processing_pipeline_name = training_job_analytics.iloc[0][
            "TrainingJobDefinitionName"
        ]

        logging.info(
            "Chosen Data Processing pipeline candidate name is {}".format(
                best_data_processing_pipeline_name
            )
        )

        best_candidate = self.candidates[best_data_processing_pipeline_name]
        return best_candidate

    # some helper function to display candidates as HTML table
    def display_candidates(self):
        from IPython.display import display, HTML

        row_format_string = (
            "<tr><th>{candidate_name}</th><td>{algo_name}</td>"
            "<td><a href='{module_root}/{dpp_name}.py'>{dpp_name}.py</a></td></tr>"
        )

        job_name = self.local_run_config.automl_job_name
        candidate_html_rows = "\n".join(
            [
                row_format_string.format(
                    candidate_name=candidate.candidate_name,
                    algo_name=candidate.algo_name,
                    module_root=f"{job_name}-artifacts/generated_module/candidate_data_processors",
                    dpp_name=candidate.data_transformer_step.name,
                )
                for candidate in self.candidates.values()
            ]
        )

        html = """
            <table>
            <tr><th>Candidate Name</th><th>Algorithm</th><th>Feature Transformer</th></tr>
            {}
            </table>
            """.format(
            candidate_html_rows
        )

        display(HTML(html))
