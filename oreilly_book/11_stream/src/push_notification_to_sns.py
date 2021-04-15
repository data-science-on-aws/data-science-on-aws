from __future__ import print_function
import boto3
import base64
import os

SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]

sns = boto3.client("sns")

print("Loading function")


def lambda_handler(event, context):
    output = []
    success = 0
    failure = 0
    highest_score = 0

    print("event: {}".format(event))
    r = event["records"]
    print("records: {}".format(r))
    print("type_records: {}".format(type(r)))

    for record in event["records"]:
        try:
            # Uncomment the below line to publish the decoded data to the SNS topic.
            payload = base64.b64decode(record["data"])
            print("payload: {}".format(payload))
            text = payload.decode("utf-8")
            print("text: {}".format(text))
            score = float(text)
            if (score != 0) and (score > highest_score):
                highest_score = score
                print("New highest_score: {}".format(highest_score))
                # sns.publish(TopicArn=SNS_TOPIC_ARN, Message='New anomaly score: {}'.format(text), Subject='New Reviews Anomaly Score Detected')
                output.append({"recordId": record["recordId"], "result": "Ok"})
                success += 1
        except Exception as e:
            print(e)
            output.append({"recordId": record["recordId"], "result": "DeliveryFailed"})
            failure += 1
    if highest_score != 0:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message="New anomaly score: {}".format(str(highest_score)),
            Subject="New Reviews Anomaly Score Detected",
        )
    print("Successfully delivered {0} records, failed to deliver {1} records".format(success, failure))
    return {"records": output}
