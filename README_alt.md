# Upcoming O'Reilly Book:  _Data Science on Amazon Web Services_
Register for early access directly on our [**website**](https://datascienceonaws.com).

Request one of our [**talks**](https://datascienceonaws.com/talks) for your conference or meetup.

_Influence the book by filling out our [**quick survey**](https://www.surveymonkey.com/r/798CMZ3)._

[![Data Science on Amazon Web Services](img/data-science-on-aws-book.png)](https://datascienceonaws.com)

# Workshop Cost
This workshop is FREE, but would otherwise cost <25 USD.

![Workshop Cost](img/billing.png)

# Workshop Agenda

![Workshop Agenda](img/outline.png)

# Workshop Instructions

## 1. Click on AWS Console

Take the defaults and click on **Open AWS Console**. This will open AWS Console in a new browser tab.

![AWS Console](img/alt_aws_console.png)

Double-check that your account name is something like `IibsAdminAccess-DO-NOT-DELETE...` as follows:

![IAM Role](img/alt_iibsadminaccess.png)

If not, please logout of your AWS Console in all browser tabs and re-run the steps above!

## 2. Create `TeamRole` IAM Role

![IAM](img/alt_iam_1.png)

![Roles](img/alt_roles_2.png)

![Create Role](img/alt_create_role_3.png)

![Select Service](img/alt_select_service_4.png)

![Select Policy](img/alt_select_policy_5.png)

![Add Tags](img/alt_add_tags_6.png)

![Review Name](img/alt_review_name_7.png)

## 3. Launch an Amazon SageMaker Notebook Instance

Open the [AWS Management Console](https://console.aws.amazon.com/console/home)

![Back to SageMaker](img/alt_back_to_sagemaker_8.png)

In the AWS Console search bar, type `SageMaker` and select `Amazon SageMaker` to open the service console.

![Notebook Instances](img/alt_notebook_instances_9.png)

![Create Notebook Part 1](img/alt_create_notebook_10.png)

In the Notebook instance name text box, enter `workshop`.

Choose `ml.t3.medium`. We'll only be using this instance to launch jobs. The training job themselves will run either on a SageMaker managed cluster or an Amazon EKS cluster.

Volume size `250` - this is needed to explore datasets, build docker containers, and more.  During training data is copied directly from Amazon S3 to the training cluster when using SageMaker.  When using Amazon EKS, we'll setup a distributed file system that worker nodes will use to get access to training data.

![Fill notebook instance](img/alt-notebook-setup01.png)

In the IAM role box, select the default `TeamRole`.

![Fill notebook instance](img/notebook-setup02.png)

You must select the default `VPC`, `Subnet`, and `Security group` as shown in the screenshow.  Your values will likely be different.  This is OK.

Keep the default settings for the other options not highlighted in red, and click `Create notebook instance`.  On the `Notebook instances` section you should see the status change from `Pending` -> `InService`

![Fill notebook instance](img/alt-notebook-setup03.png)

While the notebook spins up, continue to work on the next section.  We'll come back to the notebook when it's ready.

## 4. Update IAM Role Policy

Click on the `notebook` instance to see the instance details.
`
![Notebook Instance Details](img/alt_click_notebook_instance.png)

Click on the IAM role link and navigate to the IAM Management Console.

![IAM Role](img/alt_update_iam.png)

Click `Attach Policies`.

![IAM Policy](img/alt_view_policies.png)
              
Select `IAMFullAccess` and click on `Attach Policy`.

_Note:  Reminder that you should allow access only to the resources that you need._ 

![Attach Admin Policy](img/alt_attach_policies.png)

Confirm the Policies

![Confirm Policies](img/alt_confirm_policies.png)

## 4. Start the Jupyter notebook

_Note:  Proceed when the status of the notebook instance changes from `Pending` to `InService`._

![Start Jupyter](img/alt_start_jupyter.png)

## 5. Launch a new Terminal within the Jupyter notebook

Click `File` > `New` > `Terminal` to launch a terminal in your Jupyter instance.

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
Navigate to `01_intro/` in your Jupyter notebook and start the workshop!

![Start Workshop](img/start_workshop.png)


# Disclaimer
* The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
