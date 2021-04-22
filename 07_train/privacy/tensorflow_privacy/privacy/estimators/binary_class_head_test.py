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

# from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.estimators import binary_class_head
from tensorflow_privacy.privacy.estimators import test_utils
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer


class DPBinaryClassHeadTest(tf.test.TestCase):
  """Tests for DP-enabled binary class heads."""

  def testLoss(self):
    """Tests loss() returns per-example losses."""

    head = binary_class_head.DPBinaryClassHead()
    features = {'feature_a': np.full((4), 1.0)}
    labels = np.array([[1.0], [1.0], [1.0], [0.0]])
    logits = np.full((4, 1), 0.5)

    actual_loss = head.loss(labels, logits, features)
    expected_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)

    self.assertEqual(actual_loss.shape, [4, 1])

    if tf.executing_eagerly():
      self.assertEqual(actual_loss.shape, [4, 1])
      self.assertAllClose(actual_loss, expected_loss)
      return

    self.assertAllClose(expected_loss, self.evaluate(actual_loss))

  def testCreateTPUEstimatorSpec(self):
    """Tests that an Estimator built with this head works."""

    train_features, train_labels = test_utils.make_input_data(256, 2)
    feature_columns = []
    for key in train_features:
      feature_columns.append(tf.feature_column.numeric_column(key=key))

    head = binary_class_head.DPBinaryClassHead()
    optimizer = DPKerasSGDOptimizer(
        learning_rate=0.5,
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=2)
    model_fn = test_utils.make_model_fn(head, optimizer, feature_columns)
    classifier = tf.estimator.Estimator(model_fn=model_fn)

    classifier.train(
        input_fn=test_utils.make_input_fn(train_features, train_labels, True),
        steps=4)

    test_features, test_labels = test_utils.make_input_data(64, 2)
    classifier.evaluate(
        input_fn=test_utils.make_input_fn(test_features, test_labels, False),
        steps=4)

    predict_features, predict_labels_ = test_utils.make_input_data(64, 2)
    classifier.predict(
        input_fn=test_utils.make_input_fn(predict_features, predict_labels_,
                                          False))


if __name__ == '__main__':
  tf.test.main()
