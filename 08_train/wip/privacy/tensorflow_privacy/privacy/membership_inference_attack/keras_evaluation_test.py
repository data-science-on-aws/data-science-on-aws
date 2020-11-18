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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.keras_evaluation."""

from absl.testing import absltest

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.membership_inference_attack import keras_evaluation
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_flattened_attack_metrics


class UtilsTest(absltest.TestCase):

  def __init__(self, methodname):
    """Initialize the test class."""
    super().__init__(methodname)

    self.ntrain, self.ntest = 50, 100
    self.nclass = 5
    self.ndim = 10

    # Generate random training and test data
    self.train_data = np.random.rand(self.ntrain, self.ndim)
    self.test_data = np.random.rand(self.ntest, self.ndim)
    self.train_labels = np.random.randint(self.nclass, size=self.ntrain)
    self.test_labels = np.random.randint(self.nclass, size=self.ntest)

    self.model = tf.keras.Sequential([tf.keras.layers.Dense(self.nclass)])

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.model.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

  def test_calculate_losses(self):
    """Test calculating the loss."""
    pred, loss = keras_evaluation.calculate_losses(self.model, self.train_data,
                                                   self.train_labels)
    self.assertEqual(pred.shape, (self.ntrain, self.nclass))
    self.assertEqual(loss.shape, (self.ntrain,))

    pred, loss = keras_evaluation.calculate_losses(self.model, self.test_data,
                                                   self.test_labels)
    self.assertEqual(pred.shape, (self.ntest, self.nclass))
    self.assertEqual(loss.shape, (self.ntest,))

  def test_run_attack_on_keras_model(self):
    """Test the attack."""
    results = keras_evaluation.run_attack_on_keras_model(
        self.model,
        (self.train_data, self.train_labels),
        (self.test_data, self.test_labels),
        attack_types=[AttackType.THRESHOLD_ATTACK])
    self.assertIsInstance(results, AttackResults)
    att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
        results)
    self.assertLen(att_types, 2)
    self.assertLen(att_slices, 2)
    self.assertLen(att_metrics, 2)
    self.assertLen(att_values, 2)


if __name__ == '__main__':
  absltest.main()
