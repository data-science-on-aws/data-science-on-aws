import json
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.3.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==4.1.1'])
# Workaround for https://github.com/huggingface/tokenizers/issues/120 and
#                https://github.com/kaushaltrivedi/fast-bert/issues/174
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'tokenizers'])

import tensorflow as tf
from transformers import DistilBertTokenizer

classes=[1, 2, 3, 4, 5]

max_seq_length=64

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

###############################################
# Input Data looks like this (text/csv):
###############################################
# product_category,review_body,star_rating
# Gift Card,"i like the gift cards, if you need something quick and easy you can get the gift cards there easy to use, and you can get the amount on them you need in no time at all",5

###############################################
# Post-Training Bias Notebook:
# https://github.com/data-science-on-aws/workshop/blob/master/07_train/wip/clarify/04_Run_Post_Training_Bias_Analysis.ipynb
###############################################

def input_handler(data, context):
    transformed_instances = []

    print('DATA {}'.format(data))

    for instance in data:
        data_str = instance.decode('utf-8')
        print('DATA_STR {}'.format(data_str))
        
        tokens = tokenizer.tokenize(data_str)
        print('TOKENS {}'.format(tokens))
        
        encode_plus_tokens = tokenizer.encode_plus(data_str,
                                                   pad_to_max_length=True,
                                                   max_length=max_seq_length,
                                                   truncation=True
                                                  )

        # Convert the text-based tokens to ids from the pre-trained BERT vocabulary
        input_ids = encode_plus_tokens['input_ids']
        # Specifies which tokens BERT should pay attention to (0 or 1)
        input_mask = encode_plus_tokens['attention_mask']
        # Segment Ids are always 0 for single-sequence tasks (or 1 if two-sequence tasks)
#        segment_ids = [0] * max_seq_length
    
        transformed_instance = { 
                                 "input_ids": input_ids, 
                                 "input_mask": input_mask, 
#                                 "segment_ids": segment_ids
                               }
    
        transformed_instances.append(transformed_instance)

    print(transformed_instances)
    
    transformed_data = {"signature_name":"serving_default",
                        "instances": transformed_instances}
    print(transformed_data)

    transformed_data_json = json.dumps(transformed_data)
    print(transformed_data_json)
    
    return transformed_data_json


def output_handler(response, context):
    
    response_json = response.json()
    print('response_json: {}'.format(response_json))
    # response_json: {'predictions': [[0.655070066, 0.5060817, 0.377855629, 0.347030044, 0.554810166]]}
    
    log_probabilities = response_json["predictions"]
    print('log_probabilities: {}'.format(log_probabilities))
    # log_probabilities: [[0.655070066, 0.5060817, 0.377855629, 0.347030044, 0.554810166]]

    predicted_classes = []

    for log_probability in log_probabilities:
        softmax = tf.nn.softmax(log_probability) 
        print('softmax: {}'.format(softmax))
        # softmax: [0.23479915 0.20229805 0.17795238 0.17255059 0.21239984]
        
        predicted_class_idx = tf.argmax(softmax, axis=-1, output_type=tf.int32)
        print('predicted_class_idx: {}'.format(predicted_class_idx))
        # predicted_class_idx: 0
        
        predicted_class = classes[predicted_class_idx]
        print('predicted_class: {}'.format(predicted_class))
        # predicted_class: 1
        
        ## new code ##
        prediction_dict = {}
        prediction_dict['predicted_label'] = predicted_class
        print('prediction_dict: {}'.format(prediction_dict))
        # {'predicted_label': 5}
        
        jsonline = json.dumps(prediction_dict)
        print('jsonline: {}'.format(jsonline))
        # jsonline: {"predicted_label": 5}
        
        predicted_classes.append(jsonline)
        print('predicted_classes: {}'.format(predicted_classes))
        
    # predicted_classes_json=json.dumps('{"predicted_label": "5", "labels": ["1", "2", "3", "4", "5"], "probabilties": ["0.01", "0.05", "0.02", "0.03", "0.09"]}')
    
#     predicted_classes_json = json.dumps(predicted_classes)    
#     print('predicted_classes_json: {}'.format(predicted_classes_json))
    
    predicted_classes_jsonlines = '\n'.join(predicted_classes)
    print('predicted_classes_jsonlines: {}'.format(predicted_classes_jsonlines))
    # Hard-Coded prediction output
    # predicted_classes_json: ['{"predicted_label": 5}', '{"predicted_label": 5}', '{"predicted_label": 5}']

    response_content_type = context.accept_header
    print('response_content_type: {}'.format(response_content_type))
    
    # Sample model output
    # ["{\"predicted_label\": 3}", "{\"predicted_label\": 3}", "{\"predicted_label\": 3}"], application/jsonlines
    return predicted_classes_jsonlines, response_content_type