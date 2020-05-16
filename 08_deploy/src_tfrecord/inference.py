import json
import tensorflow as tf
from string import whitespace
from collections import namedtuple
from google.protobuf.json_format import MessageToDict

max_seq_length = 128

############################################
# Invert this logic to read the TFRecord
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      print("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
############################################

def input_handler(data, context):
    transformed_instances = []
    
    print(type(data))
    print(data)
    
    for instance in data:
        print(type(instance))
        print(instance)

#        example = tf.train.Example()
#        example.ParseFromString(instance)
#        print(example)

#        record = MessageToDict(example.features)['feature']
#        print(record)

        decoded_instance = tf.io.decode_raw(instance, tf.uint8)
        example = tf.train.Example()
        example.ParseFromString(decoded_instance)

        example_dict = MessageToDict(example.features)
        record = example_dict['feature']

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

        print(input_ids)
        print(input_mask)
        print(segment_ids)
        
        transformed_instance = {
                                 "input_ids": input_ids,
                                 "input_mask": input_mask,
                                 "segment_ids": segment_ids
                               }

        print(transformed_instance)
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


if __name__ == "__main__":
    # Read sample tfrecord file

    filenames = ['../data-tfrecord/bert-test/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.tfrecord']

    tfrecord_dataset= tf.data.TFRecordDataset(filenames)
    print(tfrecord_dataset)

    instances = []

    for tfrecord in tfrecord_dataset.take(2):
        # Decode tfrecord
        instance = input_handler(tfrecord, None)
        print(instance)
        instances.append(instance)

    print(instances)
