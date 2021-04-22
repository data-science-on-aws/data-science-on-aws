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
"""Utility functions for membership inference attacks."""

import numpy as np
import scipy.special


def log_loss(labels: np.ndarray, pred: np.ndarray, small_value=1e-8):
  """Compute the cross entropy loss.

  Args:
    labels: numpy array of shape (num_samples,) labels[i] is the true label
      (scalar) of the i-th sample
    pred: numpy array of shape(num_samples, num_classes) where pred[i] is the
      probability vector of the i-th sample
    small_value: a scalar. np.log can become -inf if the probability is too
      close to 0, so the probability is clipped below by small_value.

  Returns:
    the cross-entropy loss of each sample
  """
  return -np.log(np.maximum(pred[range(labels.size), labels], small_value))


def log_loss_from_logits(labels: np.ndarray, logits: np.ndarray):
  """Compute the cross entropy loss from logits."""
  return log_loss(labels, scipy.special.softmax(logits, axis=-1))
