import json
import tensorflow as tf
from transformers import DistilBertTokenizer

review_body_column_idx_tsv = 13

classes=[1, 2, 3, 4, 5]

max_seq_length=128

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def input_handler(data, context):
    transformed_instances = []

    for instance in data:

        data_str = instance.decode('utf-8')

        data_str_split = data_str.split('\t')
#        print(len(data_str_split))
        if (len(data_str_split) >= review_body_column_idx_tsv):
            print(data_str_split[review_body_column_idx_tsv])

        text_input = data_str_split[review_body_column_idx_tsv]
        tokens = tokenizer.tokenize(text_input)

        encode_plus_tokens = tokenizer.encode_plus(text_input,
                                                   pad_to_max_length=True,
                                                   max_length=max_seq_length)

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

    transformed_data = {"instances": transformed_instances}

    return json.dumps(transformed_data)


def output_handler(response, context):
    response_json = response.json()

#     print(type(response_json))
#     print(response_json)

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

