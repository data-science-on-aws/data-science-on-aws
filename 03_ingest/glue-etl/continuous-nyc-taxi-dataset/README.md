Use the following to create a CF deployment from the cli:

```
aws cloudformation deploy --template-file ./CF-x.json --stack-name y --capabilities CAPABILITY_IAM
```

See https://docs.aws.amazon.com/AWSCloudFormation/latest/APIReference/API_CreateStack.html for more info on flags used to create a CF deployment.

## 1/ Automated and continuous deployment of Amazon SageMaker models with AWS Step Functions
* https://aws.amazon.com/blogs/machine-learning/automated-and-continuous-deployment-of-amazon-sagemaker-models-with-aws-step-functions/
* https://github.com/aws-samples/aws-sagemaker-build

Deploy with CloudFormation:
* Original [template](https://s3.amazonaws.com/aws-machine-learning-blog/artifacts/sagebuild/v1/template.json)
* Local copy: ```CF-sagebuild.json```

**Careful! Leaves a SNS Topic 'open to public' which triggers internal Isengard Trouble Ticket and security alert.

## 2/ AIM357-2019-ETL-and-ML-Workshop
* https://github.com/bsnively/AIM357-2019-ETL-and-ML-Workshop
* https://aim357.readthedocs.io/en/latest/

Deploy with CloudFormation: 
* Original [template](https://raw.githubusercontent.com/bsnively/AIM357-2019-ETL-and-ML-Workshop/master/etl-cfn-2am-trigger.json) 
* Local copy: ```CF-aim357-etl-cfn-2am-trigger.json```


## 3/ Model Development Life Cycle (MDLC)
* [ARC340-R1 - Amazon.com automating machine learning deployments at scale](https://d1.awsstatic.com/events/reinvent/2019/REPEAT_1_Amazon.com_automating_machine_learning_deployments_at_scale_ARC340-R1.pdf)
* https://youtu.be/_mfTG63sAF0
* https://github.com/awskieran/mdlc_workshop_reinvent2019
* https://w.amazon.com/bin/view/Lending/RAP_Projects/CP_ML_Learning_Program/MDLC/

Relevant Links
* SparkMagic:  https://github.com/jupyter-incubator/sparkmagic/blob/master/README.md
