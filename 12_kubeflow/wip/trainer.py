
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from typing import Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs


_LABEL_KEY = 'label'
BERT_TFHUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def load_bert_layer(model_url=BERT_TFHUB_URL):
    # Load the pre-trained BERT model as layer in Keras
    bert_layer = hub.KerasLayer(
        handle=model_url,
        trainable=False)  # model can be fine-tuned 
    return bert_layer

def get_model(tf_transform_output, max_seq_length=64, num_labels=2):

    # dynamically create inputs for all outputs of our transform graph
    feature_spec = tf_transform_output.transformed_feature_spec()  
    feature_spec.pop(_LABEL_KEY)

    inputs = {
        key: tf.keras.layers.Input(shape=(max_seq_length), name=key, dtype=tf.int64)
            for key in feature_spec.keys()
    }

    input_word_ids = tf.cast(inputs["input_word_ids"], dtype=tf.int32)
    input_mask = tf.cast(inputs["input_mask"], dtype=tf.int32)
    input_type_ids = tf.cast(inputs["input_type_ids"], dtype=tf.int32)

    bert_layer = load_bert_layer()
    pooled_output, _ = bert_layer(
        [input_word_ids, 
         input_mask, 
         input_type_ids
        ]
    )
    
    # Add additional layers depending on your problem
    x = tf.keras.layers.Dense(256, activation='relu')(pooled_output)
    dense = tf.keras.layers.Dense(64, activation='relu')(x)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    keras_model = tf.keras.Model(
        inputs=[
                inputs['input_word_ids'], 
                inputs['input_mask'], 
                inputs['input_type_ids']], 
        outputs=pred)
    keras_model.compile(loss='binary_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                        metrics=['accuracy']
                        )
    return keras_model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn

def _input_fn(file_pattern: Text,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: input tfrecord file pattern.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_LABEL_KEY)

    return dataset

# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 32)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = get_model(tf_transform_output=tf_transform_output)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                                          tf.TensorSpec(
                                              shape=[None],
                                              dtype=tf.string,
                                              name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)