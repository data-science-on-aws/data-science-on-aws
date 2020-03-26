from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import pandas as pd
from datetime import datetime
import subprocess
import sys

# We should remove this once the bug is fixed.
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==1.15.2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-hub==0.7.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bert-tensorflow==1.0.1'])

import tensorflow as tf
import tensorflow_hub as hub

print(tf.__version__)

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re

import argparse
import json
import os
import pandas as pd
import csv
import glob
from pathlib import Path

# Based on this...
# https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L479

MAX_SEQ_LENGTH = 128
LABEL_VALUES = [1, 2, 3, 4, 5]

# TODO:  Pass this into the processor
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
# This function wraps our model function in a `model_fn_builder` function that adapts our model to work for training, evaluation, and prediction.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  def model_fn(features, labels, mode, params):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])
      
        return bert.tokenization.FullTokenizer(vocab_file=vocab_file,
                                               do_lower_case=do_lower_case)
    
    
def predict(in_sentences):
    labels = ["1", "2", "3", "4", "5"]

    tokenizer = create_tokenizer_from_hub_module()
    
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label

    input_features = run_classifier.convert_examples_to_features(input_examples, LABEL_VALUES, MAX_SEQ_LENGTH, tokenizer)

    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

    predictions = estimator.predict(predict_input_fn)

    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

#    parser.add_argument('--model-type', type=str, default='bert')
#    parser.add_argument('--model-name', type=str, default='bert-base-cased')
    parser.add_argument('--train-data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args, _ = parser.parse_known_args()   
#    model_type = args.model_type
#    model_name = args.model_name
    train_data = args.train_data
    validation_data = args.validation_data
    model_dir = args.model_dir
    hosts = args.hosts
    current_host = args.current_host
    num_gpus = args.num_gpus

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    USE_BUCKET = False 

    # Compute # train and warmup steps from batch size
    
    # We need the # of training rows.  For now, just hard-coding
    #num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_train_steps = 100
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
      num_labels=len(LABEL_VALUES),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})

    # Next we create an input builder function that takes our training feature set (`train_features`) and produces a generator. This is a pretty standard design pattern for working with Tensorflow [Estimators](https://www.tensorflow.org/guide/estimators).

    train_data_filenames = glob.glob('{}/*.tfrecord'.format(train_data))
    print(train_data_filenames)
    
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.file_based_input_fn_builder(
        train_data_filenames,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)
    
    print('Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('Training took time ', datetime.now() - current_time)
    print('Ending Training!')
        
    print('Complete')
#   TODO:  Figure out why this gets stuck    
#
#     print('Begin Validating!')
#     validation_data_filenames = glob.glob('{}/*.tfrecord'.format(validation_data))
#     print(validation_data_filenames)
#     validation_input_fn = bert.run_classifier.file_based_input_fn_builder(
#         validation_data_filenames,
#         seq_length=MAX_SEQ_LENGTH,
#         is_training=False,
#         drop_remainder=False)

#     # Now let's use our test data to see how well our model did:
#     estimator.evaluate(input_fn=validation_input_fn, steps=None)
#     print('End Validating!')
    
    # Now let's write code to make predictions on new sentences:
#     pred_sentences = [
#       "That movie was absolutely awful",
#       "The acting was a bit lacking",
#       "The film was creative and surprising",
#       "Absolutely fantastic!"
#     ]

#     print('Begin Predicting!')
#     predictions = predict(pred_sentences)
#     print(predictions)
#     print('End Predicting!')