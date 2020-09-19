# Upcoming O'Reilly Book:  _Data Science on AWS_
Register for early access directly on our [**website**](https://datascienceonaws.com).

[![Data Science on AWS](img/data-science-on-aws-book.png)](https://datascienceonaws.com)

# Workshop Instructions
_Note:  This workshop will create an ephemeral AWS acccount for each attendee.  This ephemeral account is not accessible after the workshop.  You can, of course, clone this GitHub repo and reproduce the entire workshop in your own AWS Account._


## 1. Login to the Workshop Portal (aka Event Engine). 

![Event Box Account](img/eb-aws-account-clean.png). 

![Event Box Account](img/ee-accept.png). 

![Event Engine Dashboard](img/event-engine-dashboard.png). 

## 2. Login to the **AWS Console**

![Event Engine AWS Console](img/event-engine-aws-console.png)

Take the defaults and click on **Open AWS Console**. This will open AWS Console in a new browser tab.

If you see this message, you need to logout from any previously used AWS accounts.

![AWS Account Logout](img/aws-logout.png)

_Please logout of all AWS Console sessions in all browser tabs._


Double-check that your account name is something like `TeamRole/MasterKey` as follows:

![IAM Role](img/teamrole-masterkey.png)

If not, please logout of your AWS Console in all browser tabs and re-run the steps above!


## 3. Launch a SageMaker Notebook Instance

Open the [AWS Management Console](https://console.aws.amazon.com/console/home)

**Note:** This workshop has been tested on the US West (Oregon) (us-west-2) region. Make sure that you see **Oregon** on the top right hand corner of your AWS Management Console. If you see a different region, click the dropdown menu and select US West (Oregon).

In the AWS Console search bar, type `SageMaker` and select `Amazon SageMaker` to open the service console.

![SageMaker Console](img/setup_aws_console.png). 

Select `Create notebook instance`.

![SageMaker Console](img/aws-sagemaker-dashboard.png).

![SageMaker Console](img/create-notebook-instance.png)

In the Notebook instance name text box, enter `workshop`.

Choose `ml.c5.2xlarge`. We'll only be using this instance to launch jobs. The training job themselves will run either on a SageMaker managed cluster or an Amazon EKS cluster.

Volume size `250` - this is needed to explore datasets, build docker containers, and more.  During training data is copied directly from Amazon S3 to the training cluster when using SageMaker.  When using Amazon EKS, we'll setup a distributed file system that worker nodes will use to get access to training data.

![Fill notebook instance](img/notebook-setup01.png)

In the IAM role box, select the default `TeamRole`.

![Fill notebook instance](img/notebook-setup02.png)

You must select the default `VPC`, `Subnet`, and `Security group` as shown in the screenshow.  Your values will likely be different.  This is OK.

Keep the default settings for the other options not highlighted in red, and click `Create notebook instance`.  On the `Notebook instances` section you should see the status change from `Pending` -> `InService`

![Fill notebook instance](img/notebook-setup03-olt.png)

In case you see a message like the following please delete the notebook instance and re-create the notebook instance selecting the subnet-a in the VPC settings. 

![Wrong subnet](img/wrong_subnet.png)


## 4. Start the Jupyter Notebook

_Note:  Proceed when the status of the notebook instance changes from `Pending` to `InService` after a few minutes._

![Start Jupyter](img/start_jupyter.png)


## 5. Launch a New Terminal within the Jupyter Notebook

Click `File` > `New` > [...scroll down...] `Terminal` to launch a terminal in your Jupyter instance.

![](img/launch_jupyter_terminal.png)


## 6. Clone this GitHub Repo in the Terminal

Within the Jupyter terminal, run the following:

```
cd ~/SageMaker && git clone https://github.com/data-science-on-aws/workshop
```

![](img/clone-workshop-repo.png)


## 7. Navigate Back to Notebook View

![](img/back-to-jupyter-notebook.png)


## 8. Start the Workshop!
Navigate to `01_setup/` in your Jupyter notebook and start the workshop!

_You may need to refresh your browser if you don't see the new `workshop/` directory._

![Start Workshop](img/start_workshop.png)
