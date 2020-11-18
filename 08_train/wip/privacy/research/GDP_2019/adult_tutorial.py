# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Training a one-layer NN on Adult data with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_mu_poisson
from tensorflow_privacy.privacy.optimizers import dp_optimizer

#### FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD.'
    'If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.55,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1, 'Clipping norm')
flags.DEFINE_integer('epochs', 20, 'Number of epochs')
flags.DEFINE_integer('max_mu', 2, 'GDP upper limit')
flags.DEFINE_string('model_dir', None, 'Model directory')

sampling_batch = 256
microbatches = 256
num_examples = 29305


def nn_model_fn(features, labels, mode):
  """Define CNN architecture using tf.keras.layers."""
  input_layer = tf.reshape(features['x'], [-1, 123])
  y = tf.keras.layers.Dense(16, activation='relu').apply(input_layer)
  logits = tf.keras.layers.Dense(2).apply(y)

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    if FLAGS.dpsgd:
      # Use DP version of GradientDescentOptimizer. Other optimizers are
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.train.Optimizer should be wrappable in differentially private
      # counterparts by calling dp_optimizer.optimizer_from_args().
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=microbatches,
          learning_rate=FLAGS.learning_rate)
      opt_loss = vector_loss
    else:
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
      opt_loss = scalar_loss
    global_step = tf.compat.v1.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.compat.v1.metrics.accuracy(
                labels=labels, predictions=tf.argmax(input=logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, eval_metric_ops=eval_metric_ops)

  return None


def load_adult():
  """Loads ADULT a2a as in LIBSVM and preprocesses to combine training and validation data."""
  # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

  x = pd.read_csv('adult.csv')
  kf = KFold(n_splits=10)
  for train_index, test_index in kf.split(x):
    train, test = x.iloc[train_index, :], x.iloc[test_index, :]
  train_data = train.iloc[:, range(x.shape[1] - 1)].values.astype('float32')
  test_data = test.iloc[:, range(x.shape[1] - 1)].values.astype('float32')

  train_labels = (train.iloc[:, x.shape[1] - 1] == 1).astype('int32').values
  test_labels = (test.iloc[:, x.shape[1] - 1] == 1).astype('int32').values

  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(0)

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_adult()

  # Instantiate the tf.Estimator.
  adult_classifier = tf.compat.v1.estimator.Estimator(
      model_fn=nn_model_fn, model_dir=FLAGS.model_dir)

  # Create tf.Estimator input functions for the training and test data.
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={'x': test_data}, y=test_labels, num_epochs=1, shuffle=False)

  # Training loop.
  steps_per_epoch = num_examples // sampling_batch
  test_accuracy_list = []
  for epoch in range(1, FLAGS.epochs + 1):
    for _ in range(steps_per_epoch):
      whether = np.random.random_sample(num_examples) > (
          1 - sampling_batch / num_examples)
      subsampling = [i for i in np.arange(num_examples) if whether[i]]
      global microbatches
      microbatches = len(subsampling)

      train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
          x={'x': train_data[subsampling]},
          y=train_labels[subsampling],
          batch_size=len(subsampling),
          num_epochs=1,
          shuffle=True)
      # Train the model for one step.
      adult_classifier.train(input_fn=train_input_fn, steps=1)

    # Evaluate the model and print results
    eval_results = adult_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    test_accuracy_list.append(test_accuracy)
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

    # Compute the privacy budget expended so far.
    if FLAGS.dpsgd:
      eps = compute_eps_poisson(epoch, FLAGS.noise_multiplier, num_examples,
                                sampling_batch, 1e-5)
      mu = compute_mu_poisson(epoch, FLAGS.noise_multiplier, num_examples,
                              sampling_batch)
      print('For delta=1e-5, the current epsilon is: %.2f' % eps)
      print('For delta=1e-5, the current mu is: %.2f' % mu)

      if mu > FLAGS.max_mu:
        break
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
