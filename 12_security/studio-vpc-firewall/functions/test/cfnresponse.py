# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json

SUCCESS = "SUCCESS"
FAILED = "FAILED"

def send(event, context, responseStatus, responseData, physicalResourceId=None, noEcho=False, reason=None):
    print(f"cfnresponse.send:")
    print(f"responseStatus={responseStatus}")
    print(f"responseData={json.dumps(responseData, indent=2)}")
    print(f"physicalResourceId={physicalResourceId}")
