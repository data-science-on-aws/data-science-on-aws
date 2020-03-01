#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
IMAGE="fluent-sagemaker-fast-bert"

# parameters
FASTAI_VERSION="1.0"
PY_VERSION="py36"

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

chmod +x bert/train.py
chmod +x bert/serve.py

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-eu-west-1}

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${IMAGE}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${IMAGE}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 520713654638 --region ${region} --no-include-email)

# loop for each architecture (cpu & gpu)

echo "Building image with arch=gpu, region=${region}"
TAG="pytorch-gpu-${PY_VERSION}"
FULLNAME="${account}.dkr.ecr.${region}.amazonaws.com/${IMAGE}:${TAG}"
docker build -t ${IMAGE}:${TAG} --no-cache --build-arg ARCH="gpu" -f "Dockerfile_pytorch_nvidia" .
docker tag ${IMAGE}:${TAG} ${FULLNAME}
docker push ${FULLNAME}
