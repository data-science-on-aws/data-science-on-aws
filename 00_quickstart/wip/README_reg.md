# Data Science on AWS - O'Reilly Book
[![Data Science on AWS](img/data-science-on-aws-book.png)](https://www.amazon.com/Data-Science-AWS-End-End/dp/1492079391/)

## Book Outline
![Book Outline](img/outline.png)

# Quick Start Workshop (4-hours)
![Workshop Paths](img/workshop_paths1.png)

In this quick start hands-on workshop, you will build an end-to-end AI/ML pipeline for natural language processing with Amazon SageMaker.  You will train and tune a text classifier to predict the star rating (1 is bad, 5 is good) for product reviews using the state-of-the-art [BERT](https://arxiv.org/abs/1810.04805) model for language representation.  To build our BERT-based NLP text classifier, you will use a product reviews dataset where each record contains some review text and a star rating (1-5).

## Quick Start Workshop Learning Objectives
Attendees will learn how to do the following:
* Ingest data into S3 using Amazon Athena and the Parquet data format
* Visualize data with pandas, matplotlib on SageMaker notebooks
* Detect statistical data bias with SageMaker Clarify
* Perform feature engineering on a raw dataset using Scikit-Learn and SageMaker Processing Jobs
* Store and share features using SageMaker Feature Store
* Train and evaluate a custom BERT model using TensorFlow, Keras, and SageMaker Training Jobs
* Evaluate the model using SageMaker Processing Jobs
* Track model artifacts using Amazon SageMaker ML Lineage Tracking
* Run model bias and explainability analysis with SageMaker Clarify
* Register and version models using SageMaker Model Registry
* Deploy a model to a REST endpoint using SageMaker Hosting and SageMaker Endpoints
* Automate ML workflow steps by building end-to-end model pipelines using SageMaker Pipelines

# Extended Workshop (8-hours)
![Workshop Paths](img/workshop_paths2.png)

In the extended hands-on workshop, you will get hands-on with advanced model training and deployment techniques such as hyper-parameter tuning, A/B testing, and auto-scaling.  You will also setup a real-time, streaming analytics and data science pipeline to perform window-based aggregations and anomaly detection.

## Extended Workshop Learning Objectives
Attendees will learn how to do the following:
* Perform automated machine learning (AutoML) to find the best model from just your dataset with low-code
* Find the best hyper-parameters for your custom model using SageMaker Hyper-parameter Tuning Jobs
* Deploy multiple model variants into a live, production A/B test to compare online performance, live-shift prediction traffic, and autoscale the winning variant using SageMaker Hosting and SageMaker Endpoints
* Setup a streaming analytics and continuous machine learning application using Amazon Kinesis and SageMaker

# Workshop Instructions

## 1. Login to AWS Console

![Console](img/aws_console.png)

## 2. Launch SageMaker Studio

Open the [AWS Management Console](https://console.aws.amazon.com/console/home)

Configure IAM to run the workshop.

![IAM 1](img/sagemaker-iam-1.png)

![IAM 2](img/sagemaker-iam-2.png)

![IAM 3](img/sagemaker-iam-3.png)

![Back to SageMaker](img/alt_back_to_sagemaker_8.png)

In the AWS Console search bar, type `SageMaker` and select `Amazon SageMaker` to open the service console.

![Notebook Instances](img/stu_notebook_instances_9.png)

![Quick Start](img/sm-quickstart-iam-existing.png)

![Pending Studio](img/studio_pending.png)

![Open Studio](img/studio_open.png)

![Loading Studio](img/studio_loading.png)

## 3. Launch a New Terminal within Studio

Click `File` > `New` > `Terminal` to launch a terminal in your Jupyter instance.

![Terminal Studio](img/studio_terminal.png)

## 4. Clone this GitHub Repo in the Terminal

Within the Terminal, run the following:

```
cd ~ && git clone https://github.com/data-science-on-aws/workshop
```

If you see an error like the following, just re-run the command again until it works:
```
fatal: Unable to create '/home/sagemaker-user/workshop/.git/index.lock': File exists.

Another git process seems to be running in this repository, e.g.
an editor opened by 'git commit'. Please make sure all processes
are terminated then try again. If it still fails, a git process
may have crashed in this repository earlier:
remove the file manually to continue.
```
_Note:  This is not a fatal error ^^ above ^^.  Just re-run the command again until it works._

## 5. Start the Workshop!

Navigate to `workshop/00_quickstart/` in SageMaker Studio and start the workshop!

_You may need to refresh your browser if you don't see the new `workshop/` directory._

![Start Workshop](img/studio_start_workshop.png)
