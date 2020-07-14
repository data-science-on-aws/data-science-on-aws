# Upcoming O'Reilly Book:  _Data Science on Amazon Web Services_
Register for early access directly on our [**website**](https://datascienceonaws.com).

Request one of our [**talks**](https://datascienceonaws.com/talks) for your conference or meetup.

_Influence the book by filling out our [**quick survey**](https://www.surveymonkey.com/r/798CMZ3)._

[![Data Science on Amazon Web Services](img/data-science-on-aws-book.png)](https://datascienceonaws.com)

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

![Notebook Instances](img/stu_notebook_instances_9.png)

![Create Studio](img/studio_create.png)

![Pending Studio](img/studio_pending.png)

![Open Studio](img/studio_open.png)

![Loading Studio](img/studio_loading.png)

![Terminal Studio](img/studio_terminal.png)

![Select Workshop](img/studio_select_workshop.png)

![Start Workshop](img/studio_start_workshop.png)

## 4. Update IAM Role Policy

![Select IAM](img/studio_select_iam.png)

![Select Roles](img/studio_select_roles.png)

![Edit TeamRole](img/studio_edit_teamrole.png)

Click `Attach Policies`.

![IAM Policy](img/view_policies.png)
              
Select `AmazonS3FullAccess` and click on `Attach Policy`.

_Note:  Reminder that you should allow access only to the resources that you need._ 

![Attach Admin Policy](img/alt_attach_policies.png)

## 4. Start the Jupyter notebook

_Note:  Proceed when the status of the notebook instance changes from `Pending` to `InService`._

![Start Jupyter](img/start_jupyter.png)

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
