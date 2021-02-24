# Copyright 2020, The TensorFlow Authors.
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

"""Train a CNN on MNIST with DP-SGD optimizer on TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
import mnist_dpsgd_tutorial_common as common

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.77,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 200, 'Batch size')
flags.DEFINE_integer('cores', 2, 'Number of TPU cores')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 100, 'Number of microbatches '
    '(must evenly divide batch_size / cores)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('master', None, 'Master')

FLAGS = flags.FLAGS


def cnn_model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
  """Model function for a CNN."""

  # Define CNN architecture using tf.keras.layers.
  logits = common.get_cnn_model(features)

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

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
          num_microbatches=FLAGS.microbatches,
          learning_rate=FLAGS.learning_rate)
      opt_loss = vector_loss
    else:
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
      opt_loss = scalar_loss

    # Training with TPUs requires wrapping the optimizer in a
    # CrossShardOptimizer.
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(labels, logits):
      predictions = tf.argmax(logits, 1)
      return {
          'accuracy':
              tf.metrics.accuracy(labels=labels, predictions=predictions),
      }

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=scalar_loss,
        eval_metrics=(metric_fn, {
            'labels': labels,
            'logits': logits,
        }))


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Instantiate the tf.Estimator.
  run_config = tf.estimator.tpu.RunConfig(master=FLAGS.master)
  mnist_classifier = tf.estimator.tpu.TPUEstimator(
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      model_fn=cnn_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config)

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  eval_steps_per_epoch = 10000 // FLAGS.batch_size
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    # Train the model for one epoch.
    mnist_classifier.train(
        input_fn=common.make_input_fn(
            'train', FLAGS.batch_size / FLAGS.cores, tpu=True),
        steps=steps_per_epoch)
    end_time = time.time()
    logging.info('Epoch %d time in seconds: %.2f', epoch, end_time - start_time)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        input_fn=common.make_input_fn(
            'test', FLAGS.batch_size / FLAGS.cores, 1, tpu=True),
        steps=eval_steps_per_epoch)
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

    # Compute the privacy budget expended.
    if FLAGS.dpsgd:
      if FLAGS.noise_multiplier > 0.0:
        # Due to the nature of Gaussian noise, the actual noise applied is
        # equal to FLAGS.noise_multiplier * sqrt(number of cores).
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            60000, FLAGS.batch_size,
            FLAGS.noise_multiplier * math.sqrt(FLAGS.cores), epoch, 1e-5)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
      else:
        print('Trained with DP-SGD but with zero noise.')
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
