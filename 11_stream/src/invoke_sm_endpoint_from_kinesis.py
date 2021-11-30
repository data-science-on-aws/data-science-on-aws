from __future__ import print_function

import base64
import os
import io
import boto3
import json
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONLinesSerializer
from sagemaker.deserializers import JSONLinesDeserializer

# grab environment variables
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
print("Endpoint: {}".format(ENDPOINT_NAME))

sess = sagemaker.Session()

predictor = Predictor(
    endpoint_name=ENDPOINT_NAME,
    serializer=JSONLinesSerializer(),
    deserializer=JSONLinesDeserializer(),
    sagemaker_session=sess
)


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
            
        predicted_classes = predictor.predict(inputs)
        print("predicted_classes: {}".format(predicted_classes))

        #predicted_classes_str = response["Body"].read().decode()
        #predicted_classes_json = json.loads(predicted_classes_str)
        #predicted_classes_json = json.loads(response.decode())

        #predicted_classes = predicted_classes_json.splitlines()
        #print("predicted_classes: {}".format(predicted_classes))

        for predicted_class_dict, input_data in zip(predicted_classes, inputs):
            predicted_class = predicted_class_dict["predicted_label"]
            print('Predicted star_rating: {} for review_body "{}"'.format(predicted_class, input_data["features"][0]))

            # Built output_record
            # review_id, star_rating, product_category, review_body
            output_data = "{}\t{}\t{}\t{}".format(
                split_inputs[0], str(predicted_class), split_inputs[1], input_data["features"]
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
