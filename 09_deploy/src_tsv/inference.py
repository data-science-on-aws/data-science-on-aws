import json
import tensorflow as tf
from transformers import DistilBertTokenizer

review_body_column_idx_tsv = 13

classes=[1, 2, 3, 4, 5]

max_seq_length=128

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def input_handler(data, context):
    transformed_instances = []
    print(type(data))
    print(data)

    for instance in data:
        print(type(instance))
        print(instance)

        data_str = instance.decode('utf-8')
        print(type(data_str))
        print(data_str)

        data_str_split = data_str.split('\t')
        print(len(data_str_split))
        if (len(data_str_split) >= review_body_column_idx_tsv):
            print(data_str_split[review_body_column_idx_tsv])

        tokens_a = tokenizer.tokenize(data_str_split[review_body_column_idx_tsv])

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []  
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)  
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        transformed_instance = { 
                                 "input_ids": input_ids, 
                                 "input_mask": input_mask, 
                                 "segment_ids": segment_ids
                               }
    
        transformed_instances.append(transformed_instance)

    transformed_data = {"instances": transformed_instances}
    print(transformed_data)

    return json.dumps(transformed_data)


def output_handler(response, context):
    print(type(response))
    print(response)

    response_json = response.json()

    print(type(response_json))
    print(response_json)

    log_probabilities = response_json["predictions"]

    predicted_classes = []

    for log_probability in log_probabilities:
        softmax = tf.nn.softmax(log_probability)    
        predicted_class_idx = tf.argmax(softmax, axis=-1, output_type=tf.int32)
        predicted_class = classes[predicted_class_idx]
        predicted_classes.append(predicted_class)

    print(predicted_classes)
    predicted_classes_json = json.dumps(predicted_classes)

    response_content_type = context.accept_header

    return predicted_classes_json, response_content_type

