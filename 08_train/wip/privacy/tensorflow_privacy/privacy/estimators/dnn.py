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

import tensorflow as tf

from tensorflow_privacy.privacy.estimators import head_utils
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import dnn


class DNNClassifier(tf.estimator.Estimator):
  """DP version of DNNClassifier."""

  def __init__(
      self,
      hidden_units,
      feature_columns,
      model_dir=None,
      n_classes=2,
      weight_column=None,
      label_vocabulary=None,
      optimizer=None,
      activation_fn=tf.nn.relu,
      dropout=None,
      config=None,
      warm_start_from=None,
      loss_reduction=tf.keras.losses.Reduction.NONE,
      batch_norm=False,
  ):
    head = head_utils.binary_or_multi_class_head(
        n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
    estimator._canned_estimator_api_gauge.get_cell('Classifier').set('DNN')

    def _model_fn(features, labels, mode, config):
      return dnn.dnn_model_fn_v2(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          config=config,
          batch_norm=batch_norm)

    super(DNNClassifier, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)
