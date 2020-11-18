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
"""DP Logistic Regression on MNIST.

DP Logistic Regression on MNIST with support for privacy-by-iteration analysis.
Vitaly Feldman, Ilya Mironov, Kunal Talwar, and Abhradeep Thakurta.
"Privacy amplification by iteration."
In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS),
pp. 521-532. IEEE, 2018.
https://arxiv.org/abs/1808.06651.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer

GradientDescentOptimizer = tf.train.GradientDescentOptimizer

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.05,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_integer('batch_size', 5, 'Batch size')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
flags.DEFINE_float('regularizer', 0, 'L2 regularizer coefficient')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_float('data_l2_norm', 8, 'Bound on the L2 norm of normalized data')


def lr_model_fn(features, labels, mode, nclasses, dim):
  """Model function for logistic regression."""
  input_layer = tf.reshape(features['x'], tuple([-1]) + dim)
  logits = tf.keras.layers.Dense(
      units=nclasses,
      kernel_regularizer=tf.keras.regularizers.L2(l2=FLAGS.regularizer),
      bias_regularizer=tf.keras.regularizers.L2(l2=FLAGS.regularizer)).apply(
          input_layer)

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits) + tf.losses.get_regularization_loss()
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    if FLAGS.dpsgd:
      # The loss function is L-Lipschitz with L = sqrt(2*(||x||^2 + 1)) where
      # ||x|| is the norm of the data.
      # We don't use microbatches (thus speeding up computation), since no
      # clipping is necessary due to data normalization.
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=math.sqrt(2 * (FLAGS.data_l2_norm**2 + 1)),
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=1,
          learning_rate=FLAGS.learning_rate)
      opt_loss = vector_loss
    else:
      optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(input=logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, eval_metric_ops=eval_metric_ops)


def normalize_data(data, data_l2_norm):
  """Normalizes data such that each samples has bounded L2 norm.

  Args:
    data: the dataset. Each row represents one samples.
    data_l2_norm: the target upper bound on the L2 norm.
  """

  for i in range(data.shape[0]):
    norm = np.linalg.norm(data[i])
    if norm > data_l2_norm:
      data[i] = data[i] / norm * data_l2_norm


def load_mnist(data_l2_norm=float('inf')):
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape(train_data.shape[0], -1)
  test_data = test_data.reshape(test_data.shape[0], -1)

  idx = np.random.permutation(len(train_data))  # shuffle data once
  train_data = train_data[idx]
  train_labels = train_labels[idx]

  normalize_data(train_data, data_l2_norm)
  normalize_data(test_data, data_l2_norm)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  return train_data, train_labels, test_data, test_labels


def print_privacy_guarantees(epochs, batch_size, samples, noise_multiplier):
  """Tabulating position-dependent privacy guarantees."""
  if noise_multiplier == 0:
    print('No differential privacy (additive noise is 0).')
    return

  print('In the conditions of Theorem 34 (https://arxiv.org/abs/1808.06651) '
        'the training procedure results in the following privacy guarantees.')

  print('Out of the total of {} samples:'.format(samples))

  steps_per_epoch = samples // batch_size
  orders = np.concatenate(
      [np.linspace(2, 20, num=181),
       np.linspace(20, 100, num=81)])
  delta = 1e-5
  for p in (.5, .9, .99):
    steps = math.ceil(steps_per_epoch * p)  # Steps in the last epoch.
    coef = 2 * (noise_multiplier * batch_size)**-2 * (
        # Accounting for privacy loss
        (epochs - 1) / steps_per_epoch +  # ... from all-but-last epochs
        1 / (steps_per_epoch - steps + 1))  # ... due to the last epoch
    # Using RDP accountant to compute eps. Doing computation analytically is
    # an option.
    rdp = [order * coef for order in orders]
    eps, _, _ = get_privacy_spent(orders, rdp, target_delta=delta)
    print('\t{:g}% enjoy at least ({:.2f}, {})-DP'.format(
        p * 100, eps, delta))

  # Compute privacy guarantees for the Sampled Gaussian Mechanism.
  rdp_sgm = compute_rdp(batch_size / samples, noise_multiplier,
                        epochs * steps_per_epoch, orders)
  eps_sgm, _, _ = get_privacy_spent(orders, rdp_sgm, target_delta=delta)
  print('By comparison, DP-SGD analysis for training done with the same '
        'parameters and random shuffling in each epoch guarantees '
        '({:.2f}, {})-DP for all samples.'.format(eps_sgm, delta))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.data_l2_norm <= 0:
    raise ValueError('data_l2_norm must be positive.')
  if FLAGS.dpsgd and FLAGS.learning_rate > 8 / FLAGS.data_l2_norm**2:
    raise ValueError('The amplification-by-iteration analysis requires'
                     'learning_rate <= 2 / beta, where beta is the smoothness'
                     'of the loss function and is upper bounded by ||x||^2 / 4'
                     'with ||x|| being the largest L2 norm of the samples.')

  # Load training and test data.
  # Smoothness = ||x||^2 / 4 where ||x|| is the largest L2 norm of the samples.
  # To get bounded smoothness, we normalize the data such that each sample has a
  # bounded L2 norm.
  train_data, train_labels, test_data, test_labels = load_mnist(
      data_l2_norm=FLAGS.data_l2_norm)

  # Instantiate tf.Estimator.
  # pylint: disable=g-long-lambda
  model_fn = lambda features, labels, mode: lr_model_fn(
      features, labels, mode, nclasses=10, dim=train_data.shape[1:])
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Create tf.Estimator input functions for the training and test data.
  # To analyze the per-user privacy loss, we keep the same orders of samples in
  # each epoch by setting shuffle=False.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=False)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data}, y=test_labels, num_epochs=1, shuffle=False)

  # Train the model.
  num_samples = train_data.shape[0]
  steps_per_epoch = num_samples // FLAGS.batch_size

  mnist_classifier.train(
      input_fn=train_input_fn, steps=steps_per_epoch * FLAGS.epochs)

  # Evaluate the model and print results.
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print('Test accuracy after {} epochs is: {:.2f}'.format(
      FLAGS.epochs, eval_results['accuracy']))

  if FLAGS.dpsgd:
    print_privacy_guarantees(
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        samples=num_samples,
        noise_multiplier=FLAGS.noise_multiplier,
    )

if __name__ == '__main__':
  app.run(main)
