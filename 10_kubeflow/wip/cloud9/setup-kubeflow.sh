source ~/.bash_profile

#### Install `eksctl`
# To get started we'll first install the `awscli` and `eksctl` CLI tools. [eksctl](https://eksctl.io) simplifies the process of creating EKS clusters.

pip install awscli --upgrade --user

curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp

sudo mv /tmp/eksctl /usr/local/bin

eksctl version

#### Install `kubectl`
# `kubectl` is a command line interface for running commands against Kubernetes clusters. 
# Run the following to install Kubectl

curl --location -o ./kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.15.10/2020-02-22/bin/linux/amd64/kubectl

chmod +x ./kubectl

sudo mv ./kubectl /usr/local/bin

kubectl version --short --client

#### Install `aws-iam-authenticator`

curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.15.10/2020-02-22/bin/linux/amd64/aws-iam-authenticator

chmod +x ./aws-iam-authenticator

sudo mv aws-iam-authenticator /usr/local/bin

aws-iam-authenticator version

#### Install jq and envsubst (from GNU gettext utilities) 
sudo yum -y install jq gettext

#### Verify the binaries are in the path and executable
for command in kubectl jq envsubst
  do
    which $command &>/dev/null && echo "$command in path" || echo "$command NOT FOUND"
  done

echo "Completed"


source ~/.bash_profile

export AWS_REGION=$(aws configure get region)
echo "export AWS_REGION=${AWS_REGION}" | tee -a ~/.bash_profile

export AWS_CLUSTER_NAME=cluster
echo "export AWS_CLUSTER_NAME=${AWS_CLUSTER_NAME}" | tee -a ~/.bash_profile

echo "Completed"


source ~/.bash_profile

cat << EOF > cluster.yaml
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${AWS_CLUSTER_NAME}
  region: ${AWS_REGION}

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]

managedNodeGroups:
- name: cpu-nodes
  instanceType: c5.xlarge
  volumeSize: 100
  desiredCapacity: 5
  iam:
    withAddonPolicies:
      albIngress: true

#secretsEncryption:
#  keyARN: ${MASTER_ARN}
EOF

cat cluster.yaml

echo "Completed"


source ~/.bash_profile

eksctl create cluster -f ./cluster.yaml

echo "Completed"


source ~/.bash_profile

eksctl utils describe-stacks --region=${AWS_REGION} --cluster=${AWS_CLUSTER_NAME}

### Source the environment
source ~/.bash_profile

### Create more environment variables
#export STACK_NAME=$(eksctl get nodegroup --cluster ${AWS_CLUSTER_NAME} -o json | jq -r '.[].StackName')
#echo "export STACK_NAME=${STACK_NAME}" | tee -a ~/.bash_profile

export INSTANCE_ROLE_NAME=$(aws iam list-roles \
    | jq -r ".Roles[] \
    | select(.RoleName \
    | startswith(\"eksctl-$AWS_CLUSTER_NAME\") and contains(\"NodeInstanceRole\")) \
    .RoleName")
#export INSTANCE_ROLE_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --output text --query "Stacks[0].Outputs[1].OutputValue" | sed -e 's/.*\///g')
echo "export INSTANCE_ROLE_NAME=${INSTANCE_ROLE_NAME}" | tee -a ~/.bash_profile

#export INSTANCE_PROFILE_ARN=$(aws cloudformation describe-stacks --stack-name $STACK_NAME | jq -r '.Stacks[].Outputs[] | select(.OutputKey=="InstanceProfileARN") | .OutputValue')
export INSTANCE_PROFILE_ARN=$(aws iam list-roles \
    | jq -r ".Roles[] \
    | select(.RoleName \
    | startswith(\"eksctl-$AWS_CLUSTER_NAME\") and contains(\"NodeInstanceRole\")) \
    .Arn")
echo "export INSTANCE_PROFILE_ARN=${INSTANCE_PROFILE_ARN}" | tee -a ~/.bash_profile

#### Allow Access from/to the Elastic Container Registry (ECR)
# This allows our cluster worker nodes to load custom Docker images (ie. models) from ECR.  We will load these custom Docker images in a later section.

aws iam attach-role-policy --role-name $INSTANCE_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

### Associated IAM and OIDC
# To use IAM roles for service accounts in your cluster, you must create an OIDC identity provider in the IAM console.  See https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html for more info.

eksctl utils associate-iam-oidc-provider --cluster ${AWS_CLUSTER_NAME} --approve
aws eks describe-cluster --name ${AWS_CLUSTER_NAME} --region ${AWS_REGION} --query "cluster.identity.oidc.issuer" --output text

echo "Completed"


source ~/.bash_profile

aws eks --region ${AWS_REGION} update-kubeconfig --name ${AWS_CLUSTER_NAME} 

kubectl get ns

kubectl get nodes

echo "Completed"


export S3_BUCKET=sagemaker-$(aws configure get region)-$(aws sts get-caller-identity | jq -r '.Account')
echo "export S3_BUCKET=${S3_BUCKET}" | tee -a ~/.bash_profile

# Create a new S3 bucket and upload the dataset. 
aws s3 ls s3://$S3_BUCKET || aws s3 mb s3://${S3_BUCKET}

echo "Completed"


source ~/.bash_profile

#### Check environment variables are set with valid values
echo $S3_BUCKET
echo $AWS_CLUSTER_NAME
#echo $STACK_NAME
echo $INSTANCE_ROLE_NAME
echo $INSTANCE_PROFILE_ARN

#### Download the `kfctl` CLI tool
curl --location https://github.com/kubeflow/kfctl/releases/download/v1.0.2/kfctl_v1.0.2-0-ga476281_linux.tar.gz | tar xz
sudo mv kfctl /usr/local/bin

#### Get the latest Kubeflow configuration file

# With Ingress
export CONFIG_URI='https://raw.githubusercontent.com/kubeflow/manifests/v1.0.2/kfdef/kfctl_aws.v1.0.2.yaml'
echo "export CONFIG_URI=${CONFIG_URI}" | tee -a ~/.bash_profile

#### Set Kubeflow environment variables 
export KF_NAME=${AWS_CLUSTER_NAME}
echo "export KF_NAME=${KF_NAME}" | tee -a ~/.bash_profile

#cd ~/SageMaker/kubeflow/notebooks/part-3-kubernetes

export KF_DIR=$PWD/${KF_NAME}
echo "export KF_DIR=${KF_DIR}" | tee -a ~/.bash_profile

#### Customize the configuration files
# We'll edit the configuration with the right names for the cluster and node groups before deploying Kubeflow.

mkdir -p ${KF_DIR}
cd ${KF_DIR}

curl -O ${CONFIG_URI}

export CONFIG_FILE=${KF_DIR}/kfctl_aws.v1.0.2.yaml
echo "export CONFIG_FILE=${CONFIG_FILE}" | tee -a ~/.bash_profile

sed -i.bak -e "/region: us-west-2/ a \      enablePodIamPolicy: true" ${CONFIG_FILE}
sed -i.bak -e "s@us-west-2@$AWS_REGION@" ${CONFIG_FILE}
sed -i.bak -e "s@roles:@#roles:@" ${CONFIG_FILE}
sed -i.bak -e "s@- eksctl-kubeflow-aws-nodegroup-ng-a2-NodeInstanceRole-xxxxxxx@#- eksctl-kubeflow-aws-nodegroup-ng-a2-NodeInstanceRole-xxxxxxx@" ${CONFIG_FILE}
sed -i.bak -e "s@eksctl-kubeflow-aws-nodegroup-ng-a2-NodeInstanceRole-xxxxxxx@$INSTANCE_ROLE_NAME@" ${CONFIG_FILE}
sed -i.bak -e 's/kubeflow-aws/'"$AWS_CLUSTER_NAME"'/' ${CONFIG_FILE}

#### Generate the Kubeflow installation files
cd ${KF_DIR}

rm -rf kustomize
rm -rf .cache

kfctl build -V -f ${CONFIG_FILE}

#### Deploy Kubeflow
cd ${KF_DIR}

kfctl apply -V -f ${CONFIG_FILE}

#### Remove istio-ingress in --namespace kubeflow
#kubectl delete ingress istio-ingress --namespace kubeflow

#### Deploy istio-ingress in --namespace istio-system
#cd .cache/manifests/manifests-1.0-branch/ 
#kubectl apply -k aws/istio-ingress/base --namespace istio-system
#cd ${KF_DIR}

#### Delete the usage reporting beacon
kubectl delete all -l app=spartakus --namespace=kubeflow

#### Change to `kubeflow` namespace
kubectl config set-context --current --namespace=kubeflow

kubectl -n kubeflow get all

echo "Completed"


source ~/.bash_profile

aws iam create-user --user-name s3user
aws iam attach-user-policy --user-name s3user --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-user-policy --user-name s3user --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-user-policy --user-name s3user --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

echo "Completed"


source ~/.bash_profile

aws iam create-access-key --user-name s3user > /tmp/create_output.json

echo "Completed"


source ~/.bash_profile

export AWS_ACCESS_KEY_ID_VALUE=$(jq -j .AccessKey.AccessKeyId /tmp/create_output.json | base64)
echo "export AWS_ACCESS_KEY_ID_VALUE=${AWS_ACCESS_KEY_ID_VALUE}" | tee -a ~/.bash_profile

export AWS_SECRET_ACCESS_KEY_VALUE=$(jq -j .AccessKey.SecretAccessKey /tmp/create_output.json | base64)
echo "export AWS_SECRET_ACCESS_KEY_VALUE=${AWS_SECRET_ACCESS_KEY_VALUE}" | tee -a ~/.bash_profile

#### Apply to EKS cluster.

cat <<EOF | kubectl apply --namespace anonymous -f -
apiVersion: v1
kind: Secret
metadata:
  name: aws-secret
type: Opaque
data:
  AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID_VALUE
  AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY_VALUE
EOF

#### Add the secret to the `kubeflow` namespace, as well.  This is needed until KF Pipelines support namespaces.
cat <<EOF | kubectl apply --namespace kubeflow -f -
apiVersion: v1
kind: Secret
metadata:
  name: aws-secret
type: Opaque
data:
  AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID_VALUE
  AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY_VALUE
EOF

echo "Completed"


source ~/.bash_profile

TRUST="{ \"Version\": \"2012-10-17\", \"Statement\": [ { \"Effect\": \"Allow\", \"Principal\": { \"Service\": \"sagemaker.amazonaws.com\" }, \"Action\": \"sts:AssumeRole\" } ] }"
aws iam create-role --role-name workshop-sagemaker-kfp-role --assume-role-policy-document "$TRUST"
aws iam attach-role-policy --role-name workshop-sagemaker-kfp-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name workshop-sagemaker-kfp-role --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

export SAGEMAKER_ROLE_ARN=$(aws iam get-role --role-name workshop-sagemaker-kfp-role --output text --query 'Role.Arn')
echo "export SAGEMAKER_ROLE_ARN=${SAGEMAKER_ROLE_ARN}" | tee -a ~/.bash_profile

cat <<EoF > sagemaker-invoke.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "*"
        }
    ]
}
EoF

aws iam put-role-policy --role-name workshop-sagemaker-kfp-role --policy-name sagemaker-invoke-for-worker --policy-document file://sagemaker-invoke.json
aws iam put-role-policy --role-name ${INSTANCE_ROLE_NAME} --policy-name sagemaker-invoke-for-worker --policy-document file://sagemaker-invoke.json

cat sagemaker-invoke.json

echo "Completed"


source ~/.bash_profile

aws iam attach-role-policy --role-name ${INSTANCE_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

echo "Completed"

