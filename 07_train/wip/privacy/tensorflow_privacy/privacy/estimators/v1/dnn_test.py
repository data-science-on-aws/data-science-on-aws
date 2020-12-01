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
"""Tests for DP-enabled DNNClassifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.estimators import test_utils
from tensorflow_privacy.privacy.estimators.v1 import dnn
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


class DPDNNClassifierTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for DP-enabled DNNClassifier."""

  @parameterized.named_parameters(
      ('BinaryClassDNN', 2),
      ('MultiClassDNN 3', 3),
      ('MultiClassDNN 4', 4),
  )
  def testDNN(self, n_classes):
    train_features, train_labels = test_utils.make_input_data(256, n_classes)
    feature_columns = []
    for key in train_features:
      feature_columns.append(tf.feature_column.numeric_column(key=key))

    optimizer = functools.partial(
        DPGradientDescentGaussianOptimizer,
        learning_rate=0.5,
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=1)

    classifier = dnn.DNNClassifier(
        hidden_units=[10],
        activation_fn='relu',
        feature_columns=feature_columns,
        n_classes=n_classes,
        optimizer=optimizer,
        loss_reduction=tf.losses.Reduction.NONE)

    classifier.train(
        input_fn=test_utils.make_input_fn(train_features, train_labels, True,
                                          16))

    test_features, test_labels = test_utils.make_input_data(64, n_classes)
    classifier.evaluate(
        input_fn=test_utils.make_input_fn(test_features, test_labels, False,
                                          16))

    predict_features, predict_labels = test_utils.make_input_data(64, n_classes)
    classifier.predict(
        input_fn=test_utils.make_input_fn(predict_features, predict_labels,
                                          False))

if __name__ == '__main__':
  tf.test.main()
