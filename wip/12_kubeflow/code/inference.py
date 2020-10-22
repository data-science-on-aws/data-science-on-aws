import json
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.1.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==2.8.0'])
import tensorflow as tf
from transformers import DistilBertTokenizer

classes=[1, 2, 3, 4, 5]

max_seq_length=64

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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
#                                                   truncation=True
                                                  )

        # Convert the text-based tokens to ids from the pre-trained BERT vocabulary
        input_ids = encode_plus_tokens['input_ids']
        # Specifies which tokens BERT should pay attention to (0 or 1)
        input_mask = encode_plus_tokens['attention_mask']
        # Segment Ids are always 0 for single-sequence tasks (or 1 if two-sequence tasks)
        segment_ids = [0] * max_seq_length
    
        transformed_instance = { 
                                 "input_ids": input_ids, 
                                 "input_mask": input_mask, 
                                 "segment_ids": segment_ids
                               }
    
        transformed_instances.append(transformed_instance)

    print(transformed_instances)
    
    transformed_data = {"instances": transformed_instances}
    print(transformed_data)

    transformed_data_json = json.dumps(transformed_data)
    print(transformed_data_json)
    
    return transformed_data_json


def output_handler(response, context):
    response_json = response.json()
    print('response_json: {}'.format(response_json))
    
    log_probabilities = response_json["predictions"]

    predicted_classes = []

    for log_probability in log_probabilities:
        softmax = tf.nn.softmax(log_probability)    
        predicted_class_idx = tf.argmax(softmax, axis=-1, output_type=tf.int32)
        predicted_class = classes[predicted_class_idx]
        predicted_classes.append(predicted_class)
    
    predicted_classes_json = json.dumps(predicted_classes)    
    print(predicted_classes_json)

    response_content_type = context.accept_header

    return predicted_classes_json, response_content_type

