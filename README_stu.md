# O'Reilly Book Coming Early 2021

## Data Science on AWS

YouTube Videos, Meetups, Book, and Code:  **https://datascienceonaws.com**

[![Data Science on AWS](img/data-science-on-aws-book.png)](https://datascienceonaws.com)

# Workshop Description

In this workshop, we build a natural language processing (NLP) model to classify sample Twitter comments and customer-support emails using the state-of-the-art [BERT](https://arxiv.org/abs/1810.04805) model for language representation.

To build our BERT-based NLP model, we use the [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) which contains 150+ million customer reviews from Amazon.com for the 20 year period between 1995 and 2015.  In particular, we train a classifier to predict the `star_rating` (1 is bad, 5 is good) from the `review_body` (free-form review text).

# Workshop Agenda
![Workshop Agenda](img/outline.png)

# Workshop Contributors

![Workshop Contributors](img/primary-contributors.png)

# Workshop Instructions

## 1. Login to AWS Console

![IAM](img/aws_console.png)

## 2. Create `TeamRole` IAM Role

![IAM](img/alt_iam_1.png)

![Roles](img/alt_roles_2.png)

![Create Role](img/alt_create_role_3.png)

![Select Service](img/alt_select_service_4.png)

![Select Policy](img/alt_select_policy_5.png)

![Add Tags](img/alt_add_tags_6.png)

![Review Name](img/alt_review_name_7.png)

## 3. Update IAM Role Policy

![Select IAM](img/studio_select_iam.png)

![Edit TeamRole](img/studio_edit_teamrole.png)

Click `Attach Policies`.

![IAM Policy](img/view_policies.png)
              
Select `AmazonS3FullAccess` and click on `Attach Policy`.

_Note:  Reminder that you should allow access only to the resources that you need._ 

![Attach Admin Policy](img/alt_attach_policies.png)

## 4. Launch an Amazon SageMaker Notebook Instance

Open the [AWS Management Console](https://console.aws.amazon.com/console/home)

![Back to SageMaker](img/alt_back_to_sagemaker_8.png)

In the AWS Console search bar, type `SageMaker` and select `Amazon SageMaker` to open the service console.

![Notebook Instances](img/stu_notebook_instances_9.png)

![Create Studio](img/studio_create.png)

![Pending Studio](img/studio_pending.png)

![Open Studio](img/studio_open.png)

![Loading Studio](img/studio_loading.png)

## 5. Launch a new Terminal within the Jupyter notebook

Click `File` > `New` > `Terminal` to launch a terminal in your Jupyter instance.

![Terminal Studio](img/studio_terminal.png)

## 6. Clone this GitHub Repo in the Terminal

![](img/clone-workshop-repo.png)

Within the Jupyter terminal, run the following:

```
cd ~/SageMaker && git clone https://github.com/data-science-on-aws/workshop
```

## 7. Start the Workshop!

Navigate to `01_setup/` in your Jupyter notebook and start the workshop!

_You may need to refresh your browser if you don't see the new `workshop/` directory._

![Start Workshop](img/studio_start_workshop.png)
