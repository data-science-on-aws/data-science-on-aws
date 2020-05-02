import json
import tensorflow as tf
from string import whitespace
from collections import namedtuple

max_seq_length = 128

#Context = namedtuple('Context',
#                     'model_name, model_version, method, rest_uri, grpc_uri, '
#                     'custom_attributes, request_content_type, accept_header')

def input_handler(record, context):

    print(type(record))
    print(record)

    print(type(context))
    print(context)
    
#    record = data.read()

#    example = tf.train.Example()
#    example.ParseFromString(payload)
#    example_feature = MessageToDict(example.features)['feature']


#     def select_data_and_label_from_record(record):
#         x = {
#             'input_ids': record['input_ids'],
#             'input_mask': record['input_mask'],
#             'segment_ids': record['segment_ids']
#         }

#         y = record['label_ids']

#         return (x, y)

    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        transformed_record = tf.io.parse_single_example(record, name_to_features)

        input_ids = transformed_record['input_ids'].numpy().tolist()

        input_mask = transformed_record['input_mask'].numpy().tolist()

        segment_ids = transformed_record['segment_ids'].numpy().tolist()

        return input_ids, input_mask, segment_ids

    input_ids, input_mask, segment_ids = _decode_record(record, name_to_features)

    return json.dumps({"instances": {"input_ids": input_ids,
                                     "input_mask": input_mask,
                                     "segment_ids": segment_ids}
                      })

def output_handler(data, context):
    print(type(data))
    print(data)
    
    print(type(context))
    print(context)
    
    response_content_type = context.accept_header
    # Remove whitespace from output JSON string.
    prediction = response.content.decode('utf-8').translate(dict.fromkeys(map(ord,whitespace)))

    print(type(prediction))
    print(prediction)
    
    return prediction, response_content_type


if __name__ == "__main__":
    # Read sample tfrecord file

    filenames = ['../data-tfrecord/bert-test/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.tfrecord']

    tfrecord_dataset= tf.data.TFRecordDataset(filenames)
    print(tfrecord_dataset)

    instances = []

    for tfrecord in tfrecord_dataset.take(1):
        # Decode tfrecord
        instance = input_handler(tfrecord, None)
        print(instance)
        instances.append(instance)

