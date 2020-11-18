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
"""An example for using keras_evaluation."""

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_flattened_attack_metrics
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.keras_evaluation import MembershipInferenceCallback
from tensorflow_privacy.privacy.membership_inference_attack.keras_evaluation import run_attack_on_keras_model


FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.02, 'Learning rate for training')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_string('model_dir', None, 'Model directory.')
flags.DEFINE_bool('tensorboard_merge_classifiers', False, 'If true, plot '
                  'different classifiers with the same slicing_spec and metric '
                  'in the same figure.')


def small_cnn():
  """Setup a small CNN for image classification."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

  for _ in range(3):
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(10))
  return model


def load_cifar10():
  """Loads CIFAR10 data."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  x_train = np.array(x_train, dtype=np.float32) / 255
  x_test = np.array(x_test, dtype=np.float32) / 255

  y_train = np.array(y_train, dtype=np.int32).squeeze()
  y_test = np.array(y_test, dtype=np.int32).squeeze()

  return x_train, y_train, x_test, y_test


def main(unused_argv):
  # Load training and test data.
  x_train, y_train, x_test, y_test = load_cifar10()

  # Get model, optimizer and specify loss.
  model = small_cnn()
  optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Get callback for membership inference attack.
  mia_callback = MembershipInferenceCallback(
      (x_train, y_train),
      (x_test, y_test),
      slicing_spec=SlicingSpec(entire_dataset=True, by_class=True),
      attack_types=[AttackType.THRESHOLD_ATTACK,
                    AttackType.K_NEAREST_NEIGHBORS],
      tensorboard_dir=FLAGS.model_dir,
      tensorboard_merge_classifiers=FLAGS.tensorboard_merge_classifiers)

  # Train model with Keras
  model.fit(
      x_train,
      y_train,
      epochs=FLAGS.epochs,
      validation_data=(x_test, y_test),
      batch_size=FLAGS.batch_size,
      callbacks=[mia_callback],
      verbose=2)

  print('End of training attack:')
  attack_results = run_attack_on_keras_model(
      model, (x_train, y_train), (x_test, y_test),
      slicing_spec=SlicingSpec(entire_dataset=True, by_class=True),
      attack_types=[
          AttackType.THRESHOLD_ATTACK, AttackType.K_NEAREST_NEIGHBORS
      ])
  att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
      attack_results)
  print('\n'.join(['  %s: %.4f' % (', '.join([s, t, m]), v) for t, s, m, v in
                   zip(att_types, att_slices, att_metrics, att_values)]))


if __name__ == '__main__':
  app.run(main)
