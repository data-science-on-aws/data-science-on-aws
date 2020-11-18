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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.data_structures."""
from absl.testing import absltest
import numpy as np

from tensorflow_privacy.privacy.membership_inference_attack import models
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData


class TrainedAttackerTest(absltest.TestCase):

  def test_base_attacker_train_and_predict(self):
    base_attacker = models.TrainedAttacker()
    self.assertRaises(NotImplementedError, base_attacker.train_model, [], [])
    self.assertRaises(AssertionError, base_attacker.predict, [])

  def test_predict_before_training(self):
    lr_attacker = models.LogisticRegressionAttacker()
    self.assertRaises(AssertionError, lr_attacker.predict, [])

  def test_create_attacker_data_loss_only(self):
    attack_input = AttackInputData(
        loss_train=np.array([1, 3]), loss_test=np.array([2, 4]))
    attacker_data = models.create_attacker_data(attack_input, 0.5)
    self.assertLen(attacker_data.features_test, 2)
    self.assertLen(attacker_data.features_train, 2)

  def test_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16]))
    attacker_data = models.create_attacker_data(
        attack_input, 0.25, balance=False)
    self.assertLen(attacker_data.features_test, 2)
    self.assertLen(attacker_data.features_train, 3)

    for i, feature in enumerate(attacker_data.features_train):
      self.assertLen(feature, 3)  # each feature has two logits and one loss
      expected = feature[:2] not in attack_input.logits_train
      self.assertEqual(attacker_data.is_training_labels_train[i], expected)

  def test_balanced_create_attacker_data_loss_and_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1, 2], [5, 6], [8, 9]]),
        logits_test=np.array([[10, 11], [14, 15], [17, 18]]),
        loss_train=np.array([3, 7, 10]),
        loss_test=np.array([12, 16, 19]))
    attacker_data = models.create_attacker_data(attack_input, 0.33)
    self.assertLen(attacker_data.features_test, 2)
    self.assertLen(attacker_data.features_train, 4)

    for i, feature in enumerate(attacker_data.features_train):
      self.assertLen(feature, 3)  # each feature has two logits and one loss
      expected = feature[:2] not in attack_input.logits_train
      self.assertEqual(attacker_data.is_training_labels_train[i], expected)


if __name__ == '__main__':
  absltest.main()
