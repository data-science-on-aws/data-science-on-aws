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
"""Tests for DP-enabled binary class heads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_privacy.privacy.estimators import test_utils
from tensorflow_privacy.privacy.estimators.v1 import head as head_lib
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


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
        optimizer=optimizer)

  return model_fn


class DPHeadTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for DP-enabled heads."""

  # Parameters for testing: n_classes.
  @parameterized.named_parameters(
      ('Binary', 2),
      ('MultiClass 3', 3),
      ('MultiClass 4', 4),
  )
  def testCreateTPUEstimatorSpec(self, n_classes):
    """Tests that an Estimator built with a binary head works."""

    train_features, train_labels = test_utils.make_input_data(256, n_classes)
    feature_columns = []
    for key in train_features:
      feature_columns.append(tf.feature_column.numeric_column(key=key))

    head = head_lib._binary_logistic_or_multi_class_head(
        n_classes=n_classes,
        weight_column=None,
        label_vocabulary=None,
        loss_reduction=tf.compat.v1.losses.Reduction.NONE)
    optimizer = DPGradientDescentGaussianOptimizer(
        learning_rate=0.5,
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=2)
    model_fn = make_model_fn(head, optimizer, feature_columns)
    classifier = tf.estimator.Estimator(model_fn=model_fn)

    classifier.train(
        input_fn=test_utils.make_input_fn(train_features, train_labels, True),
        steps=4)

    test_features, test_labels = test_utils.make_input_data(64, n_classes)
    classifier.evaluate(
        input_fn=test_utils.make_input_fn(test_features, test_labels, False),
        steps=4)

    predict_features, predict_labels = test_utils.make_input_data(64, n_classes)
    classifier.predict(
        input_fn=test_utils.make_input_fn(predict_features, predict_labels,
                                          False))


if __name__ == '__main__':
  tf.test.main()
