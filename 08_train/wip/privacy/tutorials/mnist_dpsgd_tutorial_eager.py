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
"""Training a CNN on MNIST in TF Eager mode with DP-SGD optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

GradientDescentOptimizer = tf.train.GradientDescentOptimizer
tf.enable_eager_execution()

flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, '
                     'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches '
                     '(must evenly divide batch_size)')

FLAGS = flags.FLAGS


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def main(_):
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Fetch the mnist data
  train, test = tf.keras.datasets.mnist.load_data()
  train_images, train_labels = train
  test_images, test_labels = test

  # Create a dataset object and batch for the training data
  dataset = tf.data.Dataset.from_tensor_slices(
      (tf.cast(train_images[..., tf.newaxis]/255, tf.float32),
       tf.cast(train_labels, tf.int64)))
  dataset = dataset.shuffle(1000).batch(FLAGS.batch_size)

  # Create a dataset object and batch for the test data
  eval_dataset = tf.data.Dataset.from_tensor_slices(
      (tf.cast(test_images[..., tf.newaxis]/255, tf.float32),
       tf.cast(test_labels, tf.int64)))
  eval_dataset = eval_dataset.batch(10000)

  # Define the model using tf.keras.layers
  mnist_model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  # Instantiate the optimizer
  if FLAGS.dpsgd:
    opt = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
  else:
    opt = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  for epoch in range(FLAGS.epochs):
    # Train the model for one epoch.
    for (_, (images, labels)) in enumerate(dataset.take(-1)):
      with tf.GradientTape(persistent=True) as gradient_tape:
        # This dummy call is needed to obtain the var list.
        logits = mnist_model(images, training=True)
        var_list = mnist_model.trainable_variables

        # In Eager mode, the optimizer takes a function that returns the loss.
        def loss_fn():
          logits = mnist_model(images, training=True)  # pylint: disable=undefined-loop-variable,cell-var-from-loop
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)  # pylint: disable=undefined-loop-variable,cell-var-from-loop
          # If training without privacy, the loss is a scalar not a vector.
          if not FLAGS.dpsgd:
            loss = tf.reduce_mean(input_tensor=loss)
          return loss

        if FLAGS.dpsgd:
          grads_and_vars = opt.compute_gradients(loss_fn, var_list,
                                                 gradient_tape=gradient_tape)
        else:
          grads_and_vars = opt.compute_gradients(loss_fn, var_list)

      opt.apply_gradients(grads_and_vars)

    # Evaluate the model and print results
    for (_, (images, labels)) in enumerate(eval_dataset.take(-1)):
      logits = mnist_model(images, training=False)
      correct_preds = tf.equal(tf.argmax(input=logits, axis=1), labels)
    test_accuracy = np.mean(correct_preds.numpy())
    print('Test accuracy after epoch %d is: %.3f' % (epoch, test_accuracy))

    # Compute the privacy budget expended so far.
    if FLAGS.dpsgd:
      eps = compute_epsilon((epoch + 1) * steps_per_epoch)
      print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    else:
      print('Trained with vanilla non-private SGD optimizer')

if __name__ == '__main__':
  app.run(main)
