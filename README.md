# O'Reilly Book Coming Early 2021

## Data Science on AWS

YouTube Videos, Meetups, Book, and Code:  **https://datascienceonaws.com**

[![Data Science on AWS](img/data-science-on-aws-book.png)](https://datascienceonaws.com)

# Workshop Description
In this workshop, we build a natural language processing (NLP) model to classify sample Twitter comments and customer-support emails using the state-of-the-art [BERT](https://arxiv.org/abs/1810.04805) model for language representation.

To build our BERT-based NLP model, we use the [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) which contains 150+ million customer reviews from Amazon.com for the 20 year period between 1995 and 2015.  In particular, we train a classifier to predict the `star_rating` (1 is bad, 5 is good) from the `review_body` (free-form review text).

# Workshop Cost
This workshop is FREE, but would otherwise cost <25 USD.
![Workshop Cost](img/billing.png)

# Workshop Description
![Workshop Agenda](img/outline.png)

# Workshop Paths

## Quick Start (All-In-One Workshop Path)
![Workshop Paths](img/workshop_paths1.png)

## Additional Workshop Paths per Persona
![Workshop Paths](img/workshop_paths2.png)

# Workshop Contributors
![Workshop Contributors](img/primary-contributors.png)

# Workshop Instructions
_Note:  This workshop will create an ephemeral AWS acccount for each attendee.  This ephemeral account is not accessible after the workshop.  You can, of course, clone this GitHub repo and reproduce the entire workshop in your own AWS Account._

## 0. Logout of All AWS Consoles Across All Browser Tabs
If you do not logout of existing AWS Consoles, things will not work properly.

![AWS Account Logout](img/aws-logout.png)

_Please logout of all AWS Console sessions in all browser tabs._

## 1. Login to the Workshop Portal (aka Event Engine). 

![Event Box Launch](img/eb1_launch.png) 

![Event Box Access AWS Account](img/eb2_access_account.png)

![Event Engine Terms and Conditions](img/ee1_terms.png)

![Event Engine Dashboard](img/ee2_team_dashboard.png)


## 2. Login to the **AWS Console**

![Event Engine AWS Console](img/ee3_open_console.png)

Take the defaults and click on **Open AWS Console**. This will open AWS Console in a new browser tab.

If you see this message, you need to logout from any previously used AWS accounts.

![AWS Account Logout](img/aws-logout.png)

_Please logout of all AWS Console sessions in all browser tabs._

Double-check that your account name is similar to `TeamRole/MasterKey` as follows:

![IAM Role](img/teamrole-masterkey.png)

If not, please logout of your AWS Console in all browser tabs and re-run the steps above!


## 3. Launch SageMaker Studio

Open the [AWS Management Console](https://console.aws.amazon.com/console/home)

![Back to SageMaker](img/console1_sagemaker.png)

In the AWS Console search bar, type `SageMaker` and select `Amazon SageMaker` to open the service console.

![SageMaker Studio](img/console2_studio.png)

![Open SageMaker Studio](img/console3_open_studio.png)

![Loading Studio](img/studio_loading.png)

## 4. Launch a New Terminal within Studio

Click `File` > `New` > `Terminal` to launch a terminal in your Jupyter instance.

![Terminal Studio](img/studio_terminal.png)

## 5. Clone this GitHub Repo in the Terminal

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

## 6. Start the Workshop!

Navigate to `00_quickstart/` in SageMaker Studio and start the workshop!

_You may need to refresh your browser if you don't see the new `workshop/` directory._

![Start Workshop](img/studio_start_workshop.png)
