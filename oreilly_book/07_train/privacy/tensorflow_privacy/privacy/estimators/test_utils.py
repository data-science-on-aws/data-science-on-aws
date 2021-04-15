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
"""Helper functions for unit tests for DP-enabled Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def make_input_data(size, classes):
  """Create raw input data for testing."""
  feature_a = np.random.normal(4, 1, (size))
  feature_b = np.random.normal(5, 0.7, (size))
  feature_c = np.random.normal(6, 2, (size))
  noise = np.random.normal(0, 30, (size))
  features = {
      'feature_a': feature_a,
      'feature_b': feature_b,
      'feature_c': feature_c,
  }

  if classes == 2:
    labels = np.array(
        np.power(feature_a, 3) + np.power(feature_b, 2) +
        np.power(feature_c, 1) + noise > 125).astype(int)
  else:
    def label_fn(x):
      if x < 110.0:
        return 0
      elif x < 140.0:
        return 1
      else:
        return 2

    labels = list(map(
        label_fn,
        np.power(feature_a, 3) + np.power(feature_b, 2) +
        np.power(feature_c, 1) + noise))

  return features, labels


def make_multilabel_input_data(size):
  """Create raw input data for testing."""
  feature_a = np.random.normal(4, 1, (size))
  feature_b = np.random.normal(5, 0.7, (size))
  feature_c = np.random.normal(6, 2, (size))
  noise_a = np.random.normal(0, 1, (size))
  noise_b = np.random.normal(0, 1, (size))
  noise_c = np.random.normal(0, 1, (size))
  features = {
      'feature_a': feature_a,
      'feature_b': feature_b,
      'feature_c': feature_c,
  }

  def label_fn(a, b, c):
    return [int(a > 4), int(b > 5), int(c > 6)]

  labels = list(
      map(label_fn, feature_a + noise_a, feature_b + noise_b,
          feature_c + noise_c))

  return features, labels


def make_input_fn(features, labels, training, batch_size=16):
  """Returns an input function suitable for an estimator."""

  def input_fn():
    """An input function for training or evaluating."""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle if in training mode.
    if training:
      dataset = dataset.shuffle(1000)

    return dataset.batch(batch_size)
  return input_fn


def make_model_fn(head, optimizer, feature_columns):
  """Constructs and returns a model_fn using supplied head."""

  def model_fn(features, labels, mode, params, config=None):  # pylint: disable=unused-argument
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    inputs = feature_layer(features)
    hidden_layer = tf.keras.layers.Dense(units=3, activation='relu')
    hidden_layer_values = hidden_layer(inputs)
    logits_layer = tf.keras.layers.Dense(
        units=head.logits_dimension, activation=None)
    logits = logits_layer(hidden_layer_values)
    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        trainable_variables=hidden_layer.trainable_weights +
        logits_layer.trainable_weights,
        optimizer=optimizer)

  return model_fn
