
### Blog post

https://aws.amazon.com/blogs/aws/amazon-sagemaker-autopilot-fully-managed-automatic-machine-learning/ 

### Setup instructions

* Log out of the AWS console if you're already logged in.

* Go to https://dashboard.eventengine.run

* Enter the hash that was provided to you, and click on "Accept terms & login"

* Click on "AWS console"

* Click on "Open AWS console"

* Make sure you're using the us-east-1 region. Do not use another region.

* Select "Amazon SageMaker" in the search box

* Go to "Notebook / Notebook instances".

* Click on "Create notebook instance".

* "Notebook instance name": type a name for your instance, e.g "aim361".

* "Notebook instance type": select ml.t3.medium. No need for anything bigger.

* "IAM role": select "Create a new role"
   * Select "Any S3 bucket".
   * Click on "Create role".

* In the "Git repositories" section:
   * Select "Clone a public Git repository" from the dropdown list.
   * In the "Git repository" box, enter: https://gitlab.com/juliensimon/aim361

* Click on "Create notebook instance", and wait until the instance is "In Service"

* Click on "Open Jupyter"

* Click on the "Lab2.ipynb" notebook and get to work :) 
