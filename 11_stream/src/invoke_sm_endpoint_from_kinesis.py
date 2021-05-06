from __future__ import print_function

import base64
import os
import io
import boto3
import json

# grab environment variables
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
print("Endpoint: {}".format(ENDPOINT_NAME))
runtime = boto3.client("runtime.sagemaker")

print("Loading function")


def lambda_handler(event, context):
    outputs = []

    r = event["records"]
    print("records: {}".format(r))
    print("type_records: {}".format(type(r)))

    # TODO:  Handle batches
    for record in event["records"]:
        print(record["recordId"])
        payload = base64.b64decode(record["data"])
        print("payload: {}".format(payload))
        text = payload.decode("utf-8")
        print("text: {}".format(text))

        # Do custom processing on the payload here
        split_inputs = text.split("\t")
        print(type(split_inputs))
        print(split_inputs)
        review_body = split_inputs[2]
        print(review_body)

        inputs = [{"features": [review_body]}]

        response = runtime.invoke_endpoint(
            EndpointName=pytorch_endpoint_name,
            ContentType="application/jsonlines",
            Accept="application/jsonlines",
            Body=json.dumps(inputs).encode("utf-8"),
        )
        print("response: {}".format(response))

        predicted_classes_str = response["Body"].read().decode()
        predicted_classes_json = json.loads(predicted_classes_str)

        predicted_classes = predicted_classes_json.splitlines()
        print("predicted_classes: {}".format(predicted_classes))

        for predicted_class_json, input_data in zip(predicted_classes, inputs):
            predicted_class = json.loads(predicted_class_json)["predicted_label"]
            print('Predicted star_rating: {} for review_body "{}"'.format(predicted_class, input_data["features"][0]))

            # Built output_record
            # review_id, star_rating, product_category, review_body
            output_data = "{}\t{}\t{}\t{}".format(
                split_inputs[0], str(predicted_class), split_inputs[1], input_data["review_body"]
            )
            print("output_data: {}".format(output_data))
            output_data_encoded = output_data.encode("utf-8")

            output_record = {
                "recordId": record["recordId"],
                "result": "Ok",
                "data": base64.b64encode(output_data_encoded).decode("utf-8"),
            }
            outputs.append(output_record)

    print("Successfully processed {} records.".format(len(event["records"])))
    print("type(output): {}".format(type(outputs)))
    print("Output Length: {} .".format(len(outputs)))

    return {"records": outputs}
