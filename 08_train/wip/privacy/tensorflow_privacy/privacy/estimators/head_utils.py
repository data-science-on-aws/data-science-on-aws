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
"""Estimator heads that allow integration with TF Privacy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_privacy.privacy.estimators.binary_class_head import DPBinaryClassHead
from tensorflow_privacy.privacy.estimators.multi_class_head import DPMultiClassHead


def binary_or_multi_class_head(n_classes, weight_column, label_vocabulary,
                               loss_reduction):
  """Creates either binary or multi-class head.

  Args:
    n_classes: Number of label classes.
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
      weight_column.normalizer_fn is applied on it to get weight tensor.
    label_vocabulary: A list of strings represents possible label values. If
      given, labels must be string type and have any value in
      `label_vocabulary`. If it is not given, that means labels are already
      encoded as integer or float within [0, 1] for `n_classes=2` and encoded as
      integer values in {0, 1,..., n_classes-1} for `n_classes`>2 . Also there
      will be errors if vocabulary is not provided and labels are string.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Defines how to
      reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.

  Returns:
    A `Head` instance.
  """
  if n_classes == 2:
    head = DPBinaryClassHead(
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
  else:
    head = DPMultiClassHead(
        n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
  return head
