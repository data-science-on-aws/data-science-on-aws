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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.utils."""
from absl.testing import absltest

import numpy as np

from tensorflow_privacy.privacy.membership_inference_attack import utils


class UtilsTest(absltest.TestCase):

  def test_log_loss(self):
    """Test computing cross-entropy loss."""
    # Test binary case with a few normal values
    pred = np.array([[0.01, 0.99], [0.1, 0.9], [0.25, 0.75], [0.5, 0.5],
                     [0.75, 0.25], [0.9, 0.1], [0.99, 0.01]])
    # Test the cases when true label (for all samples) is 0 and 1
    expected_losses = {
        0:
            np.array([
                4.60517019, 2.30258509, 1.38629436, 0.69314718, 0.28768207,
                0.10536052, 0.01005034
            ]),
        1:
            np.array([
                0.01005034, 0.10536052, 0.28768207, 0.69314718, 1.38629436,
                2.30258509, 4.60517019
            ])
    }
    for c in [0, 1]:  # true label
      y = np.ones(shape=pred.shape[0], dtype=int) * c
      loss = utils.log_loss(y, pred)
      np.testing.assert_allclose(loss, expected_losses[c], atol=1e-7)

    # Test multiclass case with a few normal values
    # (values from http://bit.ly/RJJHWA)
    pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3],
                     [0.99, 0.002, 0.008]])
    # Test the cases when true label (for all samples) is 0, 1, and 2
    expected_losses = {
        0: np.array([1.60943791, 0.51082562, 0.51082562, 0.01005034]),
        1: np.array([0.35667494, 1.60943791, 2.30258509, 6.2146081]),
        2: np.array([2.30258509, 1.60943791, 1.2039728, 4.82831374])
    }
    for c in range(3):  # true label
      y = np.ones(shape=pred.shape[0], dtype=int) * c
      loss = utils.log_loss(y, pred)
      np.testing.assert_allclose(loss, expected_losses[c], atol=1e-7)

    # Test boundary values 0 and 1
    pred = np.array([[0, 1]] * 2)
    y = np.array([0, 1])
    small_values = [1e-8, 1e-20, 1e-50]
    expected_losses = np.array([18.42068074, 46.05170186, 115.12925465])
    for i, small_value in enumerate(small_values):
      loss = utils.log_loss(y, pred, small_value)
      np.testing.assert_allclose(
          loss, np.array([expected_losses[i], 0]), atol=1e-7)

  def test_log_loss_from_logits(self):
    """Test computing cross-entropy loss from logits."""

    logits = np.array([[1, 2, 0, -1], [1, 2, 0, -1], [-1, 3, 0, 0]])
    labels = np.array([0, 3, 1])
    expected_loss = np.array([1.4401897, 3.4401897, 0.11144278])

    loss = utils.log_loss_from_logits(labels, logits)
    np.testing.assert_allclose(expected_loss, loss, atol=1e-7)


if __name__ == '__main__':
  absltest.main()
