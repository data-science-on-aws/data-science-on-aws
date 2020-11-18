# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a language model (recurrent neural network) with DP-SGD optimizer.

This tutorial uses a corpus of text from TensorFlow datasets unless a
FLAGS.data_dir is specified with the path to a directory containing two files
train.txt and test.txt corresponding to a training and test corpus.

Even though we haven't done any hyperparameter tuning, and the analytical
epsilon upper bound can't offer any strong guarantees, the benefits of training
with differential privacy can be clearly seen by examining the trained model.
In particular, such inspection can confirm that the set of training-data
examples that the model fails to learn (i.e., has high perplexity for) comprises
outliers and rare sentences outside the distribution to be learned (see examples
and a discussion in this blog post). This can be further confirmed by
testing the differentially-private model's propensity for memorization, e.g.,
using the exposure metric of https://arxiv.org/abs/1802.08232.

This example is decribed in more details in this post: https://goo.gl/UKr7vH
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.001,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('data_dir', None, 'Directory containing the PTB data.')

FLAGS = flags.FLAGS

SEQ_LEN = 80
NB_TRAIN = 45000


def rnn_model_fn(features, labels, mode):  # pylint: disable=unused-argument
  """Model function for a RNN."""

  # Define RNN architecture using tf.keras.layers.
  x = features['x']
  x = tf.reshape(x, [-1, SEQ_LEN])
  input_layer = x[:, :-1]
  input_one_hot = tf.one_hot(input_layer, 256)
  lstm = tf.keras.layers.LSTM(256, return_sequences=True).apply(input_one_hot)
  logits = tf.keras.layers.Dense(256).apply(lstm)

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=tf.cast(tf.one_hot(x[:, 1:], 256), dtype=tf.float32),
      logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    if FLAGS.dpsgd:

      ledger = privacy_ledger.PrivacyLedger(
          population_size=NB_TRAIN,
          selection_probability=(FLAGS.batch_size / NB_TRAIN))

      optimizer = dp_optimizer.DPAdamGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=FLAGS.microbatches,
          ledger=ledger,
          learning_rate=FLAGS.learning_rate,
          unroll_microbatches=True)
      opt_loss = vector_loss
    else:
      optimizer = tf.train.AdamOptimizer(
          learning_rate=FLAGS.learning_rate)
      opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=tf.cast(x[:, 1:], dtype=tf.int32),
                predictions=tf.argmax(input=logits, axis=2))
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)


def load_data():
  """Load training and validation data."""
  if not FLAGS.data_dir:
    print('FLAGS.data_dir containing train.txt and test.txt was not specified, '
          'using a substitute dataset from the tensorflow_datasets module.')
    train_dataset = tfds.load(name='lm1b/subwords8k',
                              split=tfds.Split.TRAIN,
                              batch_size=NB_TRAIN,
                              shuffle_files=True)
    test_dataset = tfds.load(name='lm1b/subwords8k',
                             split=tfds.Split.TEST,
                             batch_size=10000)
    train_data = next(iter(tfds.as_numpy(train_dataset)))
    test_data = next(iter(tfds.as_numpy(test_dataset)))
    train_data = train_data['text'].flatten()
    test_data = test_data['text'].flatten()
  else:
    train_fpath = os.path.join(FLAGS.data_dir, 'train.txt')
    test_fpath = os.path.join(FLAGS.data_dir, 'test.txt')
    train_txt = open(train_fpath).read().split()
    test_txt = open(test_fpath).read().split()
    keys = sorted(set(train_txt))
    remap = {k: i for i, k in enumerate(keys)}
    train_data = np.array([remap[x] for x in train_txt], dtype=np.uint8)
    test_data = np.array([remap[x] for x in test_txt], dtype=np.uint8)

  return train_data, test_data


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / NB_TRAIN
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because Penn TreeBank has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, test_data = load_data()

  # Instantiate the tf.Estimator.
  conf = tf.estimator.RunConfig(save_summary_steps=1000)
  lm_classifier = tf.estimator.Estimator(model_fn=rnn_model_fn,
                                         model_dir=FLAGS.model_dir,
                                         config=conf)

  # Create tf.Estimator input functions for the training and test data.
  batch_len = FLAGS.batch_size * SEQ_LEN
  train_data_end = len(train_data) - len(train_data) % batch_len
  test_data_end = len(test_data) - len(test_data) % batch_len
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data[:train_data_end]},
      batch_size=batch_len,
      num_epochs=FLAGS.epochs,
      shuffle=False)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data[:test_data_end]},
      batch_size=batch_len,
      num_epochs=1,
      shuffle=False)

  # Training loop.
  steps_per_epoch = len(train_data) // batch_len
  for epoch in range(1, FLAGS.epochs + 1):
    print('epoch', epoch)
    # Train the model for one epoch.
    lm_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

    if epoch % 5 == 0:
      name_input_fn = [('Train', train_input_fn), ('Eval', eval_input_fn)]
      for name, input_fn in name_input_fn:
        # Evaluate the model and print results
        eval_results = lm_classifier.evaluate(input_fn=input_fn)
        result_tuple = (epoch, eval_results['accuracy'], eval_results['loss'])
        print(name, 'accuracy after %d epochs is: %.3f (%.4f)' % result_tuple)

    # Compute the privacy budget expended so far.
    if FLAGS.dpsgd:
      eps = compute_epsilon(epoch * steps_per_epoch)
      print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    else:
      print('Trained with vanilla non-private SGD optimizer')

if __name__ == '__main__':
  app.run(main)
