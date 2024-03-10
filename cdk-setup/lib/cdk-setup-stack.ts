import * as cdk from "@aws-cdk/core";
import * as sagemaker from "@aws-cdk/aws-sagemaker";
import * as apigateway from "@aws-cdk/aws-apigateway";
import * as autoscaling from "@aws-cdk/aws-autoscaling";
import * as cloudwatch from "@aws-cdk/aws-cloudwatch";
import * as ec2 from "@aws-cdk/aws-ec2";
import * as iam from "@aws-cdk/aws-iam";
import * as lambda from "@aws-cdk/aws-lambda";
import * as logs from "@aws-cdk/aws-logs";
import * as s3 from "@aws-cdk/aws-s3";
import * as sns from "@aws-cdk/aws-sns";
import * as sqs from "@aws-cdk/aws-sqs";
import { IUser } from "@aws-cdk/aws-iam";

interface CdkSetupStackProps extends cdk.StackProps {
  createUser?:boolean
  userName?:string
  explicitAccessPolicy?:boolean
}
export class CdkSetupStack extends cdk.Stack {
  user:IUser;
  constructor(scope: cdk.Construct, id: string, props?: CdkSetupStackProps) {
    super(scope, id, props);

    // The code that defines your stack goes here

    const accountId = props?.env?.account!;
    // not required for the demo possibly, admin account
    const createUser = props?.createUser || true;
    const userName = props?.userName || 'EEOverlord';
    if(createUser) {
      this.user = iam.User.fromUserName(this, 'AdminUser', userName);
    } else {
      this.user = new iam.User(this, 'User', {
        userName:userName, 
      });
      const AdministratorAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess');
      this.user.addManagedPolicy(AdministratorAccess);
    }

    let managedPolicies: iam.ManagedPolicy[]|undefined = undefined;
    
     // limitation of the number of policies we can attach
    const explicitAccessPolicy = props?.explicitAccessPolicy || true;
    if (explicitAccessPolicy) {
    const iamFullAccess = new iam.ManagedPolicy(this, " IAMFullAccess", {
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "iam:*",
            "organizations:DescribeAccount",
            "organizations:DescribeOrganization",
            "organizations:DescribeOrganizationalUnit",
            "organizations:DescribePolicy",
            "organizations:ListChildren",
            "organizations:ListParents",
            "organizations:ListPoliciesForTarget",
            "organizations:ListRoots",
            "organizations:ListPolicies",
            "organizations:ListTargetsForPolicy",
          ],
          resources: ["*"],
        }),
      ],
    });

    const teamDefaultPolicy = new iam.ManagedPolicy(this, " IAMFullAccess", {
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "iam:List*",
            "iam:Get*",
            "iam:Generate*",
            "sts:GetCallerIdentity",
          ],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["iam:CreateServiceLinkedRole"],
          resources: ["arn:aws:iam::*:role/aws-service-role/*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["iam:PassRole"],
          resources: [`arn:aws:iam::${accountId}:role/TeamRole`],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.DENY,
          actions: [
            "route53domains:*",
            "ec2:ModifyReservedInstances",
            "ec2:PurchaseHostReservation",
            "ec2:PurchaseReservedInstancesOffering",
            "ec2:PurchaseScheduledInstances",
            "rds:PurchaseReservedDBInstancesOffering",
            "dynamodb:PurchaseReservedCapacityOfferings",
            "s3:PutObjectRetention",
            "s3:PutObjectLegalHold",
            "s3:BypassGovernanceRetention",
            "s3:PutBucketObjectLockConfiguration",
          ],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.DENY,
          notActions: [
            //  actions: [
            "iam:*",
            "sts:*",
            "s3:*",
            "ds:*",
            "lightsail:*",
            "quicksight:*",
            "cloudfront:*",
            "route53:*",
            "servicediscovery:*",
            "waf:*",
            "waf-regional:*",
            "wafv2:*",
            "cloudwatch:DescribeAlarms",
            "cloudwatch:PutMetricAlarm",
            "cloudwatch:DeleteAlarms",
            "cloudwatch:GetMetricStatistics",
            "ec2:Describe*",
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchGetImage",
            "ecr:BatchCheckLayerAvailability",
            "ecr:GetAuthorizationToken",
            "globalaccelerator:*",
            "acm:List*",
            "acm:Describe*",
            "kms:Describe*",
            "kms:ReEncrypt*",
            "kms:Get*",
            "kms:List*",
            "kms:CreateGrant",
            "kms:RevokeGrant",
            "directconnect:Describe*",
          ],
          resources: ["*"],
          conditions: {
            // "Condition": {
            StringNotEquals: {
              "aws:RequestedRegion": ["us-west-2", "us-east-1"],
            },
          },
        }),

        new iam.PolicyStatement({
          resources: ["*"],
          effect: iam.Effect.ALLOW,
          actions: [
            "access-analyzer:*",
            "acm-pca:*",
            "acm:*",
            "amplify:*",
            "apigateway:*",
            "application-autoscaling:*",
            "applicationautoscaling:*",
            "appmesh:*",
            "appstream:*",
            "appsync:*",
            "appconfig:*",
            "arsenal:*",
            "artifact:*",
            "athena:*",
            "autoscaling-plans:*",
            "autoscaling:*",
            "awsconnector:*",
            "backup:*",
            "backup-storage:MountCapsule",
            "batch:*",
            "ce:*",
            "cloud9:*",
            "cloudformation:*",
            "cloudfront:*",
            "cloudmap:*",
            "cloudtrail:*",
            "cloudwatch:*",
            "codebuild:*",
            "codecommit:*",
            "codedeploy:*",
            "codeguru-profiler:*",
            "codeguru-reviewer:*",
            "codepipeline:*",
            "codestar:*",
            "cognito-identity:*",
            "cognito-idp:*",
            "cognito-sync:*",
            "comprehend:*",
            "comprehendmedical:*",
            "config:*",
            "dataexchange:*",
            "datapipeline:*",
            "datasync:*",
            "deepracer:*",
            "dms:*",
            "ds:*",
            "dynamodb:*",
            "ec2-instance-connect:SendSSHPublicKey",
            "ec2:*",
            "ec2messages:*",
            "ecr:*",
            "ecs:*",
            "eks:*",
            "elastic-inference:*",
            "elasticache:*",
            "elasticbeanstalk:*",
            "elasticfilesystem:*",
            "elasticloadbalancing:*",
            "elasticmapreduce:*",
            "emr:*",
            "es:*",
            "events:*",
            "execute-api:*",
            "firehose:*",
            "forecast:*",
            "fsx:*",
            "gamelift:*",
            "glacier:*",
            "globalaccelerator:*",
            "glue:*",
            "greengrass:*",
            "guardduty:*",
            "iam:*",
            "imagebuilder:*",
            "inspector:*",
            "iot:*",
            "freertos:*",
            "signer:*",
            "iotanalytics:*",
            "iotevents:*",
            "iotthingsgraph:*",
            "kafka:*",
            "kendra:*",
            "kinesis:*",
            "kinesisanalytics:*",
            "kinesisanalyticsv2:*",
            "kinesisvideo:*",
            "kms:*",
            "lakeformation:*",
            "lambda:*",
            "lex:*",
            "lightsail:*",
            "logs:*",
            "macie2:*",
            "managedblockchain:*",
            "mediaconnect:*",
            "mediaconvert:*",
            "medialive:*",
            "mediapackage-vod:*",
            "mediapackage:*",
            "mediastore:*",
            "mediatailor:*",
            "mgh:*",
            "mobiletargeting:*",
            "mq:*",
            "personalize:*",
            "pi:*",
            "pinpoint:*",
            "polly:*",
            "pricing:*",
            "qldb:*",
            "quicksight:*",
            "rds-data:*",
            "rds-db:*",
            "rds:*",
            "redshift:*",
            "rekognition:*",
            "resource-groups:*",
            "resource-explorer:*",
            "robomaker:*",
            "route53:*",
            "route53domains:DisableDomainAutoRenew",
            "route53domains:ListDomains",
            "route53resolver:*",
            "s3:*",
            "sagemaker:*",
            "schemas:*",
            "secretsmanager:*",
            "securityhub:*",
            "serverlessrepo:*",
            "servicecatalog:*",
            "servicediscovery:*",
            "ses:*",
            "sesv2:*",
            "sms:*",
            "sns:*",
            "sqs:*",
            "ssm:*",
            "ssmmessages:*",
            "states:*",
            "storagegateway:*",
            "sts:*",
            "tag:*",
            "textract:*",
            "transcribe:*",
            "transfer:*",
            "translate:*",
            "trustedadvisor:*",
            "waf-regional:*",
            "waf:*",
            "wafv2:*",
            "xray:*",
          ],
        }),

        new iam.PolicyStatement({
          resources: ["arn:aws:ec2:*:*:instance/*"],
          effect: iam.Effect.DENY,
          actions: ["ec2:RunInstances"],
          conditions: {
            StringLike: {
              "ec2:InstanceType": [
                "*6xlarge",
                "*8xlarge",
                "*10xlarge",
                "*12xlarge",
                "*16xlarge",
                "*18xlarge",
                "*24xlarge",
                "f1.4xlarge",
                "x1*",
                "z1*",
                "*metal",
              ],
            },
          },
        }),
        new iam.PolicyStatement({
          resources: ["*"],
          effect: iam.Effect.DENY,
          actions: [
            "ec2:ModifyReservedInstances",
            "ec2:PurchaseHostReservation",
            "ec2:PurchaseReservedInstancesOffering",
            "ec2:PurchaseScheduledInstances",
            "rds:PurchaseReservedDBInstancesOffering",
            "dynamodb:PurchaseReservedCapacityOfferings",
          ],
          conditions: {
            Resource: "*",
            Action: [
              "ec2:ModifyReservedInstances",
              "ec2:PurchaseHostReservation",
              "ec2:PurchaseReservedInstancesOffering",
              "ec2:PurchaseScheduledInstances",
              "rds:PurchaseReservedDBInstancesOffering",
              "dynamodb:PurchaseReservedCapacityOfferings",
            ],
            Effect: "Deny",
            Sid: "DontBuyReservationsPlz",
          },
        }),
        new iam.PolicyStatement({
          resources: [`arn:aws:iam::${accountId}:role/TeamRole`],
          effect: iam.Effect.ALLOW,
          actions: ["iam:PassRole"],
        }),
      ],
    });
    managedPolicies = [iamFullAccess, teamDefaultPolicy];

  } else {

  }
    const role = new iam.Role(this, "Role", {
      assumedBy:new iam.CompositePrincipal(
          new iam.ArnPrincipal(this.user.userArn),
          new iam.ServicePrincipal('lambda.amazonaws.com'),
          new iam.ServicePrincipal('glue.amazonaws.com'),
          new iam.ServicePrincipal('cloudwatch.amazonaws.com'),
          new iam.ServicePrincipal('ec2.amazonaws.com'),
          new iam.ServicePrincipal('ecs.amazonaws.com'),
          new iam.ServicePrincipal('deepracer.amazonaws.com'),
          new iam.ServicePrincipal('rds.amazonaws.com'),
          new iam.ServicePrincipal('dynamodb.amazonaws.com'),
          new iam.ServicePrincipal('amplify.amazonaws.com'),
          new iam.ServicePrincipal('eks.amazonaws.com'),
          new iam.ServicePrincipal('quicksight.amazonaws.com'),
          new iam.ServicePrincipal('sagemaker.amazonaws.com'),
          new iam.ServicePrincipal('cloudtrail.amazonaws.com'),
          new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
          new iam.ServicePrincipal('codecommit.amazonaws.com'),
          new iam.ServicePrincipal('robomaker.amazonaws.com'),
          new iam.ServicePrincipal('cloud9.amazonaws.com'),
      ),
      managedPolicies,
      roleName:'TeamRole',
    });

    // workshop 1 attempts to attach these anyway
    const AmazonSageMakerFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess');
    role.addManagedPolicy(AmazonSageMakerFullAccess);
    const AdministratorAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess');
    role.addManagedPolicy(AdministratorAccess);
    const IAMFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('IAMFullAccess');
    role.addManagedPolicy(IAMFullAccess);
    const AmazonS3FullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess');
    role.addManagedPolicy(AmazonS3FullAccess);
    const ComprehendFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('ComprehendFullAccess');
    role.addManagedPolicy(ComprehendFullAccess);
    const AmazonAthenaFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonAthenaFullAccess');
    role.addManagedPolicy(AmazonAthenaFullAccess);
    const SecretsManagerReadWrite = iam.ManagedPolicy.fromAwsManagedPolicyName('SecretsManagerReadWrite');
    role.addManagedPolicy(SecretsManagerReadWrite);
    const AmazonRedshiftFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonRedshiftFullAccess');
    role.addManagedPolicy(AmazonRedshiftFullAccess);

    if(!explicitAccessPolicy) {
      const AmazonEC2ContainerRegistryFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryFullAccess');
      role.addManagedPolicy(AmazonEC2ContainerRegistryFullAccess);
      const AWSStepFunctionsFullAccess = iam.ManagedPolicy.fromAwsManagedPolicyName('AWSStepFunctionsFullAccess');
      role.addManagedPolicy(AWSStepFunctionsFullAccess);
    }

    const roleArn = role.roleArn;
    const instance = new sagemaker.CfnNotebookInstance(this, "Instance", {
      instanceType: "ml.c5.2xlarge",
      roleArn,
      volumeSizeInGb: 250,
      notebookInstanceName: "workshop",
      // subnetId,
      // securityGroupIds,

    });
  }
}
