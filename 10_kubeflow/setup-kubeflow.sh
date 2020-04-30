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
export STACK_NAME=$(eksctl get nodegroup --cluster ${AWS_CLUSTER_NAME} -o json | jq -r '.[].StackName')
echo "export STACK_NAME=${STACK_NAME}" | tee -a ~/.bash_profile

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

