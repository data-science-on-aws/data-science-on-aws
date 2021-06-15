import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.3.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])
# Workaround for https://github.com/huggingface/tokenizers/issues/120 and
#                https://github.com/kaushaltrivedi/fast-bert/issues/174
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'tokenizers'])

import tensorflow as tf
from transformers import DistilBertTokenizer

classes = [1, 2, 3, 4, 5]

max_seq_length = 64

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def input_handler(data, context):
    data_str = data.read().decode("utf-8")
    print("data_str: {}".format(data_str))
    print("type data_str: {}".format(type(data_str)))

    jsonlines = data_str.split("\n")
    print("jsonlines: {}".format(jsonlines))
    print("type jsonlines: {}".format(type(jsonlines)))

    transformed_instances = []

    for jsonline in jsonlines:
        print("jsonline: {}".format(jsonline))
        print("type jsonline: {}".format(type(jsonline)))

        # features[0] is review_body
        # features[1..n] are others (ie. 1: product_category, etc)
        review_body = json.loads(jsonline)["features"][0]
        print("""review_body: {}""".format(review_body))

        encode_plus_tokens = tokenizer.encode_plus(
            review_body, pad_to_max_length=True, max_length=max_seq_length, truncation=True
        )

        # Convert the text-based tokens to ids from the pre-trained BERT vocabulary
        input_ids = encode_plus_tokens["input_ids"]

        # Specifies which tokens BERT should pay attention to (0 or 1)
        input_mask = encode_plus_tokens["attention_mask"]

        transformed_instance = {"input_ids": input_ids, "input_mask": input_mask}

        transformed_instances.append(transformed_instance)

    transformed_data = {"signature_name": "serving_default", "instances": transformed_instances}

    transformed_data_json = json.dumps(transformed_data)
    print("transformed_data_json: {}".format(transformed_data_json))

    return transformed_data_json


def output_handler(response, context):
    print("response: {}".format(response))
    response_json = response.json()
    print("response_json: {}".format(response_json))

    outputs_list = response_json["predictions"]
    print("outputs_list: {}".format(outputs_list))

    predicted_classes = []

    for outputs in outputs_list:
        print("outputs in loop: {}".format(outputs))
        print("type(outputs) in loop: {}".format(type(outputs)))

        predicted_class_idx = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        predicted_class = classes[predicted_class_idx]
        print("predicted_class: {}".format(predicted_class))

        prediction_dict = {}
        prediction_dict["predicted_label"] = predicted_class

        jsonline = json.dumps(prediction_dict)
        print("jsonline: {}".format(jsonline))

        predicted_classes.append(jsonline)
        print("predicted_classes in the loop: {}".format(predicted_classes))

    predicted_classes_jsonlines = "\n".join(predicted_classes)
    print("predicted_classes_jsonlines: {}".format(predicted_classes_jsonlines))

    response_content_type = context.accept_header

    return predicted_classes_jsonlines, response_content_type
