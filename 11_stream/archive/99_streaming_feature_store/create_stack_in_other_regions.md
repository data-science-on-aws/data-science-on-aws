# Creating the CloudFormation stack in other regions

Due to a current [restriction](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-lambda-function-code.html) in CloudFormation, the source code zip files for a Lambda function or a Lambda layer must reside in the SAME region as the STACK which you are creating. Our public copy of the zip files is in us-east-1. The instructions below provide the steps for creating the stack in regions other than us-east-1.

1. Create a new bucket in your target region to hold the artifacts needed for  executing the CloudFormation template
2. Populate that S3 bucket with input from our public bucket. Use the following S3 sync command, simply replacing `<bucket-name>` with the name of your new bucket.
````
    aws s3 sync s3://aws-ml-blog/artifacts/Using-streaming-ingestion-with-Amazon-SageMaker-Feature-Store/ s3://<bucket-name>/artifacts/Using-streaming-ingestion-with-Amazon-SageMaker-Feature-Store
````
3. Make your own “launch stack” URL, replacing the region (e.g., "eu-west-1") and the bucket name:
````
https://<REGION-NAME>.console.aws.amazon.com/cloudformation/home
?region=<REGION-NAME>#/stacks/create/template
?stackName=sm-fs-streaming-agg-stack
&templateURL=https://<BUCKET-NAME>.s3.<REGION-NAME>.amazonaws.com/
artifacts/Using-streaming-ingestion-with-Amazon-SageMaker-Feature-Store/
sagemaker-featurestore-template.yaml
````
4. Visit the final URL from step 3 in your browser.
5. Click "Next".
6. Change the first parameter (in the "Required Parameters" section) to be your bucket name (instead of "aws-ml-blog", the default).
7. Proceed as usual from there to create the stack.
