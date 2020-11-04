Based on this repo:  https://github.com/aws-samples/sagemaker-ml-workflow-with-apache-airflow

# Build End-to-End Machine Learning (ML) Workflows with Amazon SageMaker and Apache Airflow

This repository contains the assets for the Amazon Sagemaker and Apache Airflow integration sample described in this [ML blog post](#TODO).

## Overview

This repository shows a sample example to build, manage and orchestrate ML workflows using Amazon Sagemaker and Apache Airflow. We will build a recommender system to predict a customer's rating for a certain video based on customer's historical ratings of similar videos as well as the behavior of other similar customers. We'll use historical star ratings from over 2M Amazon customers on over 160K digital videos. More details on this dataset can be found at its [AWS Public Datasets page](https://s3.amazonaws.com/amazon-reviews-pds/readme.html).

### Repository Structure

The repository contains

- [AWS CloudFormation Template](./cfn/airflow-ec2.yaml) to launch the AWS services required to create the components
- [Airflow DAG Python Script](./src/dag_ml_pipeline_amazon_video_reviews.py) that integrates and orchestrates all the ML tasks in a ML workflow for building a recommender system.
- A companion [Jupyter Notebook](./notebooks/amazon-video-recommender_using_fm_algo.ipynb) to understand the individual ML tasks in detail such as data exploration, data preparation, model training/tuning and inference.


```text
.
├── README.md                                         About the repository
├── cfn                                               AWS CloudFormation Templates
│   └── airflow-ec2.yaml                              CloudFormation for installing Airflow instance backed by RDS
├── notebooks                                         Jupyter Notebooks
│   └── amazon-video-recommender_using_fm_algo.ipynb
└── src                                               Source code for Airflow DAG definition
    ├── config.py                                     Config file to configure SageMaker jobs and other ML tasks
    ├── dag_ml_pipeline_amazon_video_reviews.py       Airflow DAG definition for ML workflow
    └── pipeline                                      Python module used in Airflow DAG for data preparation
        ├── __init__.py
        ├── prepare.py                                Data preparation script
        └── preprocess.py                             Data pre-processing script
```

### High Level Solution

Here is the high-level depiction of the ML workflow we will implement for building the recommender system

![airflow_dag_workflow](./images/airflow-sagemaker-airflow-dag.png)

The workflow performs the following tasks

1. **Data Pre-processing:** Extract and pre-process data from S3 to prepare the training data.
2. **Prepare Training Data:** To build the recommender system, we will use SageMaker's built-in algorithm - Factorization machines. The algorithm expects training data only in RecordIO Protobuf format with Float32 tensors. In this task, pre-processed data will be transformed to RecordIO Protobuf format.
3. **Training the Model:** Train the SageMaker's built-in Factorization Machine model with the training data and generate model artifacts. The training job will be launched by the Airflow SageMaker operator `SageMakerTrainingOperator`.
4. **Tune the Model Hyper-parameters:** A conditional/optional task to tune the hyper-parameters of Factorization Machine to find the best model. The hyper-parameter tuning job will be launched by the SageMaker Airflow operator `SageMakerTuningOperator`.
5. **Batch inference:** Using the trained model, get inferences on the test dataset stored in Amazon S3 using Airflow SageMaker operator `SageMakerTransformOperator`.

### CloudFormation Template Resources

We will set up a simple Airflow architecture with scheduler, worker and web server running on the same instance. Typically, you will not use this setup for production workloads. We will use AWS CloudFormation to launch the AWS services required to create the components in the blog post. The stack includes the following

- Amazon EC2 instance to set up the Airflow components
- Amazon RDS (Relational Database Service) Postgres instance to host Airflow metadata database
- Amazon S3 bucket to store the Sagemaker model artifacts, outputs and Airflow DAG with ML workflow. Template will prompt for the S3 bucket name
- AWS IAM roles and EC2 Security Groups to allow Airflow components interact with the metadata database, S3 bucket and Amazon SageMaker

The prerequisite for running this CloudFormation script is to set up an Amazon EC2 Key Pair to log in to manage Airflow such as any troubleshooting or adding custom operators etc.

[![cfn-launch-stack](./images/LaunchStack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=airflow-sagemaker&templateURL=./cfn/airflow-ec2.yaml)

It may take up to 10 minutes for the CloudFormation stack to create the resources. After the resource creation is completed, you should be able to login to Airflow Web UI. The Airflow web server should be running on port 8080 by default. To open the Airflow Web UI, open any browser and type in the http://ec2-public-dns-name:8080. The public DNS Name of the EC2 instance can be found on the Outputs tab of CloudFormation stack on AWS console.

### Airflow DAG for ML Workflow

Airflow DAG integrates all the ML tasks in a ML workflow. Airflow DAG is a python script where you express individual tasks as Airflow operators, set task dependencies and associate the tasks to the DAG to run either on demand or scheduled interval. The Airflow DAG script is divided into following sections

1. Set DAG with parameters such as schedule_interval to run the workflow at scheduled time
2. Set up training, tuning and inference configurations for each operators using Sagemaker Python SDK for Airflow operators. 
3. Create individual tasks as Airflow operators defining trigger rules and associating them with the DAG object. Refer previous section for defining the individual tasks
4. Specify task dependencies

![airflow_dag](./images/airflow-sagemaker-dag.png)

You can find the Airflow DAG code [here](./src/dag_ml_pipeline_amazon_video_reviews.py) in the repo.

### Cleaning Up the Stack Resources

The final step is to clean up. To avoid unnecessary charges,

1. You should destroy all of the resources created by the CloudFormation stack in Airflow set up by deleting the stack after you’re done experimenting with it. You can follow the steps here to [delete the stack](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-delete-stack.html). 
2. You have to manually [delete the S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/delete-bucket.html) created because AWS CloudFormation cannot delete non-empty S3 bucket.

## References

- Refer [SageMaker SDK documentation](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/workflow/README.rst) and [Airflow documentation](https://airflow.apache.org/integration.html?highlight=sagemaker#amazon-sagemaker) for additional details on the Airflow SageMaker operators.
- Refer [SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html) to know more about Factorization Machines algorithm used in the blog post.

## License Summary

This sample code is made available under a modified MIT license. See the [LICENSE](./LICENSE) file.
