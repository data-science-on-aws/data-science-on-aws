
# Build, train & debug, and deploy & monitor with Amazon SageMaker

## Introduction

Amazon SageMaker is a fully managed service that removes the heavy lifting from each step of the machine learning workflow, and provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. In this interactive workshop, we will work on the different aspects of the ML workflow to build, train, and deploy a model using all the capabilities of Amazon SageMaker including the ones that we announced at re:Invent 2019. We will use the Amazon SageMaker to build, train & debug models with Amazon SageMaker Debugger, and deploy & monitor with Amazon SageMaker Model Monitor. Let’s build together!


## Datasets

In this workshop, we will go through the steps of training, debugging, deploying and monitoring a **network traffic classification model**.

For training our model we will be using datasets <a href="https://registry.opendata.aws/cse-cic-ids2018/">CSE-CIC-IDS2018</a> by CIC and ISCX which are used for security testing and malware prevention.
These datasets include a huge amount of raw network traffic logs, plus pre-processed data where network connections have been reconstructed and relevant features have been extracted using CICFlowMeter, a tool that outputs network connection features as CSV files. Each record is classified as benign traffic, or it can be malicious traffic, with a total number of 15 classes.

The goal is to demonstrate how to execute training of a network traffic classification model using the Amazon SageMaker framework container for XGBoost, training and debugging. Once trained how to then deploy and monitor the model performance.


## Getting started

Initially have an open AWS account, with privileges to create and run Amazon SageMaker notebooks and access to S3 buckets.

You can run this workshop in all commercial AWS regions where Amazon SageMaker is GA.

### Create a managed Jupyter Notebook instance
First, let's create an Amazon SageMaker managed Jupyter notebook instance.
An **Amazon SageMaker notebook instance** is a fully managed ML compute instance running the <a href="http://jupyter.org/">**Jupyter Notebook**</a> application. Amazon SageMaker manages creating the instance and related resources. 

1. In the AWS Management Console, click on Services, type “SageMaker” and press enter.
	
	<img src="images/search_sagemaker.png" alt="Search SageMaker" width="700px" />
2. You’ll be placed in the Amazon SageMaker dashboard. Click on **Notebook instances** either in the landing page or in the left menu.
	
	<img src="images/sagemaker_dashboard.png" alt="SageMaker dashboard" width="700px" />
	
3. Once in the Notebook instances screen, click on the top-righ button **Create notebook instance**.

	<img src="images/notebook_instances_screen.png" alt="Notebook Instances screen" width="700px" />
 
4. In the **Create notebook instance** screen

	<img src="images/create_notebook_instance_screen.png" alt="Create Notebook Instance screen" width="700px" />

	1. Give the Notebook Instance a name like _aim362-workshop_ or what you prefer

	2. Choose **ml.t2.medium** as **Notebook instance type**
	3. In the **IAM role** dropdown list you need to select an AWS IAM Role that is configured with security policies allowing access to Amazon SageMaker (full access) and Amazon S3 (default SageMaker buckets). If you don't have any role with those privileges, choose **Create New Role** and configure the role as follows:
	
		<img src="images/create_notebook_instance_role.png" alt="Create Notebook Instance Role" width="600px" />

	4. Keep **No VPC** selected in the **VPC** dropdown list
	5. Keep **No configuration** selected in the **Lifecycle configuration** dropdown list
	6. Keep **No Custom Encryption** selected in the **Encryption key** dropdown list
	7. Finally, click on **Create notebook instance**

4. You will be redirected to the **Notebook instances** screen and you will see a new notebook instance in _Pending_ state.

	<img src="images/notebook_instance_pending.png" alt="Notebook instance pending" width="700px" />
	
	Wait until the notebook instance is status is _In Service_ and then click on the **Open Jupyter Lab** button to be redirected to Jupyter Lab.

	<img src="images/notebook_instance_in_service.png" alt="Notebook instance in service" width="700px" />
	
	The Jupyter Lab interface will load, as shown below.
	
	<img src="images/jupyter_lab_screen.png" alt="Jupyter Lab screen" width="700px" />

### Download workshop code to the notebook instance

All the code of this workshop is implemented and available for download from this GitHub repository.

As a consequence, in this section we will clone the GitHub repository into the Amazon SageMaker notebook instance and access the Jupyter Notebooks to run the workshop.

1. From the file menu, click on **New > Terminal**
	
	<img src="images/jupyter_new_terminal.png" alt="Jupyter New Terminal tab" width="500px" />

	This will open a terminal tab in the Jupyter Lab interface
	
	<img src="images/jupyter_terminal_tab.png" alt="Jupyter Terminal Tab" width="700px" />

2. Execute the following commands in the terminal

	```
	cd SageMaker/
	git clone https://github.com/aws-samples/reinvent2019-aim362-sagemaker-debugger-model-monitor.git
	```

3. When the clone operation completes, the folder **reinvent2019-aim362-sagemaker-debugger-model-monitor** will appear automatically in the file browser on the left (if not, you can hit the **Refresh** button)

	<img src="images/jupyter_clone.png" alt="Jupyter Cloned Workshop Screen" width="700px" />
	
4. Browse to the folder **01\_train\_and\_debug** and open the file **train\_and\_debug.ipynb** to get started.

## Modules

This workshops consists of 2 modules:

- <a href="01_train_and_debug/">**01\_train\_and\_debug**</a> - Train and debug with Amazon SageMaker Debugger
- <a href="02_deploy_and_monitor/">**02\_deploy\_and\_monitor**</a> - Deploy and Monitor with Amazon SageMaker Model Monitor

You must comply with the order of modules, since the outputs of a module are inputs of the following one.


## License

The contents of this workshop are licensed under the [Apache 2.0 License](./LICENSE).

## Authors

[Giuseppe A. Porcelli](https://it.linkedin.com/in/giuporcelli) - Principal, ML Specialist Solutions Architect - Amazon Web Services EMEA<br />
[Paul Armstrong](https://www.linkedin.com/in/paul-armstrong-532bb41) - Principal Solutions Architect - Amazon Web Services EMEA
