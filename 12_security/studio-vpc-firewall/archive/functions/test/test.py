# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import time
import sagemakerdomain
import userprofile

# Data setup
DomainName = "sagemaker-anfw-domain-us-west-2"
UserProfileName = "anfw-user-profile-us-west-2"
VpcId = "vpc-0c25c864656aabc20"
SageMakerStudioSubnetIds = "subnet-08e7d40bfef63e5ea"
SageMakerSecurityGroupIds = "sg-04cf8c6953e01b74c"
SageMakerExecutionRoleArn = "arn:aws:iam::ACCOUNT_ID:role/sagemaker-studio-vpc-us-west-2-notebook-role"

UserProfileName = "demouser-profile-us-west-2"

DefaultUserSettings = {
    "SecurityGroups":SageMakerSecurityGroupIds.split(","),
    "ExecutionRole":SageMakerExecutionRoleArn
}

UserSettings = {
    "ExecutionRole":SageMakerExecutionRoleArn
}

event = {
    "RequestType":"",
    "PhysicalResourceId":"",
    "ResourceProperties":{}
}

context = None

# Test metadata
SAGEMAKER_DOMAIN = "sagemakerdomain"
USER_PROFILE = "userprofile"
tests = [SAGEMAKER_DOMAIN, USER_PROFILE]

function_map = {
    tests[0]:sagemakerdomain.handler,
    tests[1]:userprofile.handler
}

# Test driver
def test_function(test_name, f, event, context):
    print(f"Testing {test_name}\n****START****")
    print(f"Event:{event['RequestType']}")
    print(f"PhysicalResourceId:{json.dumps(event['PhysicalResourceId'], indent=2)}")
    print(f"ResourceProperties:{json.dumps(event['ResourceProperties'], indent=2)}")

    f(event, context)
    print(f"Testing {f.__name__}****END****")

def test_run(test, event_name, domain_id):
    event["RequestType"] = event_name

    if test == "sagemakerdomain":

        # SageMakerDomain test
        event["ResourceProperties"] = {
            "DomainName":DomainName,
            "VpcId":VpcId,
            "SageMakerStudioSubnetIds":SageMakerStudioSubnetIds,
            "DefaultUserSettings":DefaultUserSettings
        }
        event["PhysicalResourceId"] = domain_id

    elif test == "userprofile":

        # UserProfile test
        event["ResourceProperties"] = {
            "DomainId":domain_id,
            "UserProfileName":UserProfileName,
            "UserSettings":UserSettings
        }
        event["PhysicalResourceId"] = UserProfileName

    if event["RequestType"] == "Create":
        event["PhysicalResourceId"] = None

    test_function(test, function_map[test], event, context)

############################################
# Test
############################################

started_at = time.monotonic()

DomainId = "<domain_id>"
for t in tests:
    test_run(t, "Create", DomainId)
print(f"Create time elapsed: {time.monotonic() - started_at}")

for t in tests:
    test_run(t, "Update", DomainId)
print(f"Update time elapsed: {time.monotonic() - started_at}")

for t in tests[::-1]:
    test_run(t, "Delete", DomainId)
print(f"Delete time elapsed: {time.monotonic() - started_at}")

test_time = time.monotonic() - started_at

print(f"Time elapsed: {test_time}")

