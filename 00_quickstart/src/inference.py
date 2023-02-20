import json
import sys
import logging
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###################################
### VARIABLES
###################################

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1"])
import torch

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1"])
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import os
        
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


###################################
### SAGEMKAER LOAD MODEL FUNCTION
###################################

# You need to put in config.json from saved fine-tuned Hugging Face model in code/
# Reference it in the inference container at /opt/ml/model/code
# The model needs to be called 'model.pth' per https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/6936c08581e26ff3bac26824b1e4946ec68ffc85/src/sagemaker_pytorch_serving_container/torchserve.py#L45


def model_fn(model_dir):
    for root, dirs, files in os.walk(model_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    print(model)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

    return model


###################################
### SAGEMKAER PREDICT FUNCTION
###################################


def predict_fn(input_data, model):
    model.eval()

    print("input_data: {}".format(input_data))
    print("type(input_data): {}".format(type(input_data)))

    data_str = input_data.decode("utf-8")
    print("data_str: {}".format(data_str))
    print("type data_str: {}".format(type(data_str)))

    jsonlines = data_str.split("\n")
    print("jsonlines: {}".format(jsonlines))
    print("type jsonlines: {}".format(type(jsonlines)))

    predictions = []

    for jsonline in jsonlines:
        print("jsonline: {}".format(jsonline))
        print("type jsonline: {}".format(type(jsonline)))

        # features[0]:  review_body
        # features[1..n]:  is anything else (we can define the order ourselves)
        # Example:
        #    {"features": ["The best gift ever", "Gift Cards"]}
        #
        review_body = json.loads(jsonline)["features"][0]
        print("""review_body: {}""".format(review_body))

        result_length = 100
        inputs = tokenizer(review_body, return_tensors='pt')

        output = tokenizer.decode(model.generate(inputs["input_ids"],
                               max_length=result_length, 
                               do_sample=True, 
                               top_k=50, 
                               top_p=0.9
                              )[0])
        
        print("output: {}".format(output))

        prediction_dict = {}
        prediction_dict["generated_response"] = output

        jsonline = json.dumps(prediction_dict)
        print("jsonline: {}".format(jsonline))

        predictions.append(jsonline)
        print("predictions in the loop: {}".format(predictions))

    predictions_jsonlines = "\n".join(predictions)
    print("predictions_jsonlines: {}".format(predictions_jsonlines))

    return predictions_jsonlines


###################################
### SAGEMKAER MODEL INPUT FUNCTION
###################################

def input_fn(serialized_input_data, content_type="application/jsonlines"):
    return serialized_input_data


###################################
### SAGEMKAER MODEL OUTPUT FUNCTION
###################################

def output_fn(prediction_output, accept="application/jsonlines"):
    return prediction_output, accept
