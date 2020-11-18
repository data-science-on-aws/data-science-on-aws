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

# Lint as: python3
"""An example for using tf_estimator_evaluation."""

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_flattened_attack_metrics
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.tf_estimator_evaluation import MembershipInferenceTrainingHook
from tensorflow_privacy.privacy.membership_inference_attack.tf_estimator_evaluation import run_attack_on_tf_estimator_model


FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.02, 'Learning rate for training')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_string('model_dir', None, 'Model directory.')
flags.DEFINE_bool('tensorboard_merge_classifiers', False, 'If true, plot '
                  'different classifiers with the same slicing_spec and metric '
                  'in the same figure.')


def small_cnn_fn(features, labels, mode):
  """Setup a small CNN for image classification."""
  input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
  for _ in range(3):
    y = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    y = tf.keras.layers.MaxPool2D()(y)

  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(64, activation='relu')(y)
  logits = tf.keras.layers.Dense(10)(y)

  if mode != tf.estimator.ModeKeys.PREDICT:
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate,
                                           momentum=0.9)
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=scalar_loss, global_step=global_step)
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

  # Output the prediction probability (for PREDICT mode).
  elif mode == tf.estimator.ModeKeys.PREDICT:
    predictions = tf.nn.softmax(logits)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def load_cifar10():
  """Loads CIFAR10 data."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  x_train = np.array(x_train, dtype=np.float32) / 255
  x_test = np.array(x_test, dtype=np.float32) / 255

  y_train = np.array(y_train, dtype=np.int32).squeeze()
  y_test = np.array(y_test, dtype=np.int32).squeeze()

  return x_train, y_train, x_test, y_test


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.ERROR)
  logging.set_verbosity(logging.ERROR)
  logging.set_stderrthreshold(logging.ERROR)
  logging.get_absl_handler().use_absl_log_file()

  # Load training and test data.
  x_train, y_train, x_test, y_test = load_cifar10()

  # Instantiate the tf.Estimator.
  mnist_classifier = tf.estimator.Estimator(
      model_fn=small_cnn_fn, model_dir=FLAGS.model_dir)

  # A function to construct input_fn given (data, label), to be used by the
  # membership inference training hook.
  def input_fn_constructor(x, y):
    return tf.estimator.inputs.numpy_input_fn(x={'x': x}, y=y, shuffle=False)

  # Get hook for membership inference attack.
  mia_hook = MembershipInferenceTrainingHook(
      mnist_classifier,
      (x_train, y_train),
      (x_test, y_test),
      input_fn_constructor,
      slicing_spec=SlicingSpec(entire_dataset=True, by_class=True),
      attack_types=[AttackType.THRESHOLD_ATTACK,
                    AttackType.K_NEAREST_NEIGHBORS],
      tensorboard_dir=FLAGS.model_dir,
      tensorboard_merge_classifiers=FLAGS.tensorboard_merge_classifiers)

  # Create tf.Estimator input functions for the training and test data.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train},
      y=y_train,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_test}, y=y_test, num_epochs=1, shuffle=False)

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  for epoch in range(1, FLAGS.epochs + 1):
    # Train the model, with the membership inference hook.
    mnist_classifier.train(
        input_fn=train_input_fn, steps=steps_per_epoch, hooks=[mia_hook])

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

  print('End of training attack')
  attack_results = run_attack_on_tf_estimator_model(
      mnist_classifier, (x_train, y_train), (x_test, y_test),
      input_fn_constructor,
      slicing_spec=SlicingSpec(entire_dataset=True, by_class=True),
      attack_types=[AttackType.THRESHOLD_ATTACK, AttackType.K_NEAREST_NEIGHBORS]
      )
  att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
      attack_results)
  print('\n'.join(['  %s: %.4f' % (', '.join([s, t, m]), v) for t, s, m, v in
                   zip(att_types, att_slices, att_metrics, att_values)]))


if __name__ == '__main__':
  app.run(main)
