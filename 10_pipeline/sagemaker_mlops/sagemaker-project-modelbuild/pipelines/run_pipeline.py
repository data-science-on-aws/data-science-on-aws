# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import sys
import time

from pipelines._utils import get_pipeline_driver, convert_struct

import smexperiments
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments import tracker

import boto3

sm = boto3.Session().client(service_name="sagemaker")

import sagemaker


def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser("Creates or updates and runs the pipeline for the pipeline script.")

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        upsert_response = pipeline.upsert(role_arn=args.role_arn, description=args.description, tags=tags)
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        execution = pipeline.start()
        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

        # Now we describe execution instance and list the steps in the execution to find out more about the execution.
        execution_run = execution.describe()
        print(execution_run)

        # Create or Load the 'Experiment'
        try:
            experiment = Experiment.create(
                experiment_name=pipeline.name, description="Amazon Customer Reviews BERT Pipeline Experiment"
            )
        except:
            experiment = Experiment.load(experiment_name=pipeline.name)

        print("Experiment name: {}".format(experiment.experiment_name))

        # Add Execution Run as Trial to Experiments
        execution_run_name = execution_run["PipelineExecutionDisplayName"]
        print(execution_run_name)

        # Create the `Trial`
        timestamp = int(time.time())

        trial = Trial.create(
            trial_name=execution_run_name, experiment_name=experiment.experiment_name, sagemaker_boto_client=sm
        )

        trial_name = trial.trial_name
        print("Trial name: {}".format(trial_name))

        ######################################################
        ## Parse Pipeline Definition For Processing Job Args
        ######################################################

        processing_param_dict = {}

        for step in parsed["Steps"]:
            print("step: {}".format(step))
            if step["Name"] == "Processing":
                print("Step Name is Processing...")
                arg_list = step["Arguments"]["AppSpecification"]["ContainerArguments"]
                print(arg_list)
                num_args = len(arg_list)
                print(num_args)

                # arguments are (key, value) pairs in this list, so we extract them in pairs
                # using [i] and [i+1] indexes and stepping by 2 through the list
                for i in range(0, num_args, 2):
                    key = arg_list[i].replace("--", "")
                    value = arg_list[i + 1]
                    print("arg key: {}".format(key))
                    print("arg value: {}".format(value))
                    processing_param_dict[key] = value

        ##############################
        ## Wait For Execution To Finish
        ##############################

        print("Waiting for the execution to finish...")
        execution.wait()
        print("\n#####Execution completed. Execution step details:")

        # List Execution Steps
        print(execution.list_steps())

        # List All Artifacts Generated By The Pipeline
        processing_job_name = None
        training_job_name = None

        from sagemaker.lineage.visualizer import LineageTableVisualizer

        viz = LineageTableVisualizer(sagemaker.session.Session())
        for execution_step in reversed(execution.list_steps()):
            print(execution_step)
            # We are doing this because there appears to be a bug of this LineageTableVisualizer handling the Processing Step
            if execution_step["StepName"] == "Processing":
                processing_job_name = execution_step["Metadata"]["ProcessingJob"]["Arn"].split("/")[-1]
                print(processing_job_name)
                # display(viz.show(processing_job_name=processing_job_name))
            elif execution_step["StepName"] == "Train":
                training_job_name = execution_step["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
                print(training_job_name)
                # display(viz.show(training_job_name=training_job_name))
            else:
                # display(viz.show(pipeline_execution_step=execution_step))
                time.sleep(5)

        # Add Trial Compontents To Experiment Trial
        processing_job_tc = "{}-aws-processing-job".format(processing_job_name)
        print(processing_job_tc)

        # -aws-processing-job is the default name assigned by ProcessingJob
        response = sm.associate_trial_component(TrialComponentName=processing_job_tc, TrialName=trial_name)

        # -aws-training-job is the default name assigned by TrainingJob
        training_job_tc = "{}-aws-training-job".format(training_job_name)
        print(training_job_tc)

        response = sm.associate_trial_component(TrialComponentName=training_job_tc, TrialName=trial_name)

        ##############
        # Log Additional Parameters within Trial
        ##############
        print("Logging Processing Job Parameters within Experiment Trial...")
        processing_job_tracker = tracker.Tracker.load(trial_component_name=processing_job_tc)

        for key, value in processing_param_dict.items():
            print("key: {}, value: {}".format(key, value))
            processing_job_tracker.log_parameters({key: str(value)})
            # must save after logging
            processing_job_tracker.trial_component.save()

    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
