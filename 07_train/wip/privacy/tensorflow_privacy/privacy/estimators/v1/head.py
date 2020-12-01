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
"""Estimator v1 heads that allow integration with TF Privacy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys

# Collect together all protected access items needed from base head.
# pylint: disable=protected-access
_DEFAULT_SERVING_KEY = head_lib._DEFAULT_SERVING_KEY
_CLASSIFY_SERVING_KEY = head_lib._CLASSIFY_SERVING_KEY
_REGRESS_SERVING_KEY = head_lib._REGRESS_SERVING_KEY
_PREDICT_SERVING_KEY = head_lib._PREDICT_SERVING_KEY

_all_class_ids = head_lib._all_class_ids
_all_classes = head_lib._all_classes
_append_update_ops = head_lib._append_update_ops
_check_logits_final_dim = head_lib._check_logits_final_dim
_classification_output = head_lib._classification_output
_create_eval_metrics_tuple = head_lib._create_eval_metrics_tuple
_summary_key = head_lib._summary_key
_validate_loss_fn_args = head_lib._validate_loss_fn_args

_BaseBinaryLogisticHeadWithSigmoidCrossEntropyLoss = head_lib._BinaryLogisticHeadWithSigmoidCrossEntropyLoss
_BaseMultiClassHeadWithSoftmaxCrossEntropyLoss = head_lib._MultiClassHeadWithSoftmaxCrossEntropyLoss
# pylint: enable=protected-access


def _multi_class_head_with_softmax_cross_entropy_loss(
    n_classes,
    weight_column=None,
    label_vocabulary=None,
    loss_reduction=tf.compat.v1.losses.Reduction.SUM,
    loss_fn=None,
    name=None):
  """See `tensorflow_estimator/python/estimator/canned/head.py`."""

  if label_vocabulary is not None and not isinstance(label_vocabulary,
                                                     (list, tuple)):
    raise ValueError(
        'label_vocabulary should be a list or a tuple. Given type: {}'.format(
            type(label_vocabulary)))
  if loss_reduction not in tf.compat.v1.losses.Reduction.all():
    raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
  if loss_fn:
    _validate_loss_fn_args(loss_fn)
  return _MultiClassHeadWithSoftmaxCrossEntropyLoss(
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


class _MultiClassHeadWithSoftmaxCrossEntropyLoss(
    _BaseMultiClassHeadWithSoftmaxCrossEntropyLoss):
  """See `_multi_class_head_with_softmax_cross_entropy_loss`."""

  def _create_tpu_estimator_spec(self,
                                 features,
                                 mode,
                                 logits,
                                 labels=None,
                                 optimizer=None,
                                 train_op_fn=None,
                                 regularization_losses=None):
    """Returns a `model_fn._TPUEstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels integer or string `Tensor` with shape matching `logits`,
        namely `[D0, D1, ... DN, 1]` or `[D0, D1, ... DN]`. `labels` is required
        argument when `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` when creating the head to avoid
        scaling errors.

    Returns:
      A `model_fn._TPUEstimatorSpec` instance.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    with tf.compat.v1.name_scope(self._name, 'head'):
      logits = _check_logits_final_dim(logits, self.logits_dimension)

      # Predict.
      pred_keys = prediction_keys.PredictionKeys
      with tf.compat.v1.name_scope(None, 'predictions', (logits,)):
        all_class_ids = _all_class_ids(logits, self._n_classes)
        all_classes = _all_classes(
            logits, self._n_classes, label_vocabulary=self._label_vocabulary)
        # class_ids's shape is [D0, D1, ... DN].
        class_ids = tf.compat.v1.math.argmax(
            logits, axis=-1, name=pred_keys.CLASS_IDS)
        class_ids = tf.compat.v1.expand_dims(class_ids, axis=-1)
        if self._label_vocabulary:
          table = lookup_ops.index_to_string_table_from_tensor(
              vocabulary_list=self._label_vocabulary,
              name='class_string_lookup')
          classes = table.lookup(class_ids)
        else:
          classes = tf.strings.as_string(class_ids, name='str_classes')

        probabilities = tf.compat.v1.nn.softmax(
            logits, name=pred_keys.PROBABILITIES)
        predictions = {
            pred_keys.LOGITS: logits,
            pred_keys.PROBABILITIES: probabilities,
            # Expand to [batch_size, 1]
            pred_keys.CLASS_IDS: class_ids,
            pred_keys.CLASSES: classes,
            pred_keys.ALL_CLASS_IDS: all_class_ids,
            pred_keys.ALL_CLASSES: all_classes,
        }
      if mode == ModeKeys.PREDICT:
        classifier_output = _classification_output(
            scores=probabilities,
            n_classes=self._n_classes,
            label_vocabulary=self._label_vocabulary)
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                _DEFAULT_SERVING_KEY: classifier_output,
                _CLASSIFY_SERVING_KEY: classifier_output,
                _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)
            })

      training_loss, unreduced_loss, weights, label_ids = self.create_loss(
          features=features, mode=mode, logits=logits, labels=labels)
      if regularization_losses:
        regularization_loss = tf.math.add_n(regularization_losses)
        regularized_training_loss = tf.math.add_n(
            [training_loss, regularization_loss])
      else:
        regularization_loss = None
        regularized_training_loss = training_loss

      if self._loss_reduction == tf.compat.v1.losses.Reduction.NONE:
        scalar_loss = tf.reduce_mean(regularized_training_loss)
      else:
        scalar_loss = regularized_training_loss

      # Eval.
      if mode == ModeKeys.EVAL:
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=ModeKeys.EVAL,
            predictions=predictions,
            loss=scalar_loss,
            eval_metrics=_create_eval_metrics_tuple(
                self._eval_metric_ops, {
                    'labels': label_ids,
                    'class_ids': class_ids,
                    'weights': weights,
                    'unreduced_loss': unreduced_loss,
                    'regularization_loss': regularization_loss
                }))

      # Train.
      if optimizer is not None:
        if train_op_fn is not None:
          raise ValueError('train_op_fn and optimizer cannot both be set.')
        train_op = optimizer.minimize(
            regularized_training_loss,
            global_step=tf.compat.v1.train.get_global_step())
      elif train_op_fn is not None:
        train_op = train_op_fn(regularized_training_loss)
      else:
        raise ValueError('train_op_fn and optimizer cannot both be None.')
      train_op = _append_update_ops(train_op)
      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == tf.compat.v1.losses.Reduction.SUM:
        example_weight_sum = tf.math.reduce_sum(
            weights * tf.compat.v1.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None
    with tf.compat.v1.name_scope(''):
      keys = metric_keys.MetricKeys
      tf.compat.v1.summary.scalar(
          _summary_key(self._name, keys.LOSS), scalar_loss)
      if mean_loss is not None:
        tf.compat.v1.summary.scalar(
            _summary_key(self._name, keys.LOSS_MEAN), mean_loss)
      if regularization_loss is not None:
        tf.compat.v1.summary.scalar(
            _summary_key(self._name, keys.LOSS_REGULARIZATION),
            regularization_loss)
    return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
        mode=ModeKeys.TRAIN,
        predictions=predictions,
        loss=scalar_loss,
        train_op=train_op)


def _binary_logistic_head_with_sigmoid_cross_entropy_loss(
    weight_column=None,
    thresholds=None,
    label_vocabulary=None,
    loss_reduction=tf.compat.v1.losses.Reduction.SUM,
    loss_fn=None,
    name=None):
  """See `tensorflow_estimator/python/estimator/canned/head.py`."""

  thresholds = tuple(thresholds) if thresholds else tuple()
  if label_vocabulary is not None and not isinstance(label_vocabulary,
                                                     (list, tuple)):
    raise TypeError(
        'label_vocabulary should be a list or tuple. Given type: {}'.format(
            type(label_vocabulary)))

  for threshold in thresholds:
    if (threshold <= 0.0) or (threshold >= 1.0):
      raise ValueError('thresholds not in (0, 1): {}.'.format((thresholds,)))
  if loss_reduction not in tf.compat.v1.losses.Reduction.all():
    raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
  if loss_fn:
    _validate_loss_fn_args(loss_fn)
  return _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(
      weight_column=weight_column,
      thresholds=thresholds,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


class _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(
    _BaseBinaryLogisticHeadWithSigmoidCrossEntropyLoss):
  """DP version of `_BinaryLogisticHeadWithSigmoidCrossEntropyLoss`."""

  def _create_tpu_estimator_spec(self,
                                 features,
                                 mode,
                                 logits,
                                 labels=None,
                                 optimizer=None,
                                 train_op_fn=None,
                                 regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, 1]`. For many
        applications, the shape is `[batch_size, 1]`.
      labels: Labels integer or string `Tensor` with shape matching `logits`,
        namely `[D0, D1, ... DN, 1]` or `[D0, D1, ... DN]`. `labels` is required
        argument when `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` when creating the head to avoid
        scaling errors.

    Returns:
      `EstimatorSpec`.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    # Predict.
    with tf.compat.v1.name_scope(self._name, 'head'):
      with tf.compat.v1.name_scope(None, 'predictions', (logits,)):
        pred_keys = prediction_keys.PredictionKeys
        logits = _check_logits_final_dim(logits, self.logits_dimension)
        logistic = tf.math.sigmoid(logits, name=pred_keys.LOGISTIC)
        two_class_logits = tf.concat((tf.compat.v1.zeros_like(logits), logits),
                                     axis=-1,
                                     name='two_class_logits')
        probabilities = tf.compat.v1.nn.softmax(
            two_class_logits, name=pred_keys.PROBABILITIES)
        class_ids = tf.compat.v1.math.argmax(
            two_class_logits, axis=-1, name=pred_keys.CLASS_IDS)
        class_ids = tf.compat.v1.expand_dims(class_ids, axis=-1)
        all_class_ids = _all_class_ids(logits, n_classes=2)
        all_classes = _all_classes(
            logits, n_classes=2, label_vocabulary=self._label_vocabulary)

        if self._label_vocabulary:
          table = lookup_ops.index_to_string_table_from_tensor(
              vocabulary_list=self._label_vocabulary,
              name='class_string_lookup')
          classes = table.lookup(class_ids)
        else:
          classes = tf.strings.as_string(class_ids, name='str_classes')
        predictions = {
            pred_keys.LOGITS: logits,
            pred_keys.LOGISTIC: logistic,
            pred_keys.PROBABILITIES: probabilities,
            pred_keys.CLASS_IDS: class_ids,
            pred_keys.CLASSES: classes,
            pred_keys.ALL_CLASS_IDS: all_class_ids,
            pred_keys.ALL_CLASSES: all_classes,
        }
      if mode == ModeKeys.PREDICT:
        classifier_output = _classification_output(
            scores=probabilities,
            n_classes=2,
            label_vocabulary=self._label_vocabulary)
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                _DEFAULT_SERVING_KEY: classifier_output,
                _CLASSIFY_SERVING_KEY: classifier_output,
                _REGRESS_SERVING_KEY: export_output.RegressionOutput(
                    value=logistic),
                _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)
            })

      (training_loss, unreduced_loss, weights, processed_labels) = (
          self.create_loss(
              features=features, mode=mode, logits=logits, labels=labels))
      if regularization_losses:
        regularization_loss = tf.math.add_n(regularization_losses)
        regularized_training_loss = tf.math.add_n(
            [training_loss, regularization_loss])
      else:
        regularization_loss = None
        regularized_training_loss = training_loss

      if self._loss_reduction == tf.compat.v1.losses.Reduction.NONE:
        scalar_loss = tf.reduce_mean(regularized_training_loss)
      else:
        scalar_loss = regularized_training_loss
      # Eval.
      if mode == ModeKeys.EVAL:
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=ModeKeys.EVAL,
            predictions=predictions,
            loss=scalar_loss,
            eval_metrics=_create_eval_metrics_tuple(
                self._eval_metric_ops, {
                    'labels': processed_labels,
                    'logits': logits,
                    'logistic': logistic,
                    'class_ids': class_ids,
                    'weights': weights,
                    'unreduced_loss': unreduced_loss,
                    'regularization_loss': regularization_loss
                }))

      # Train.
      if optimizer is not None:
        if train_op_fn is not None:
          raise ValueError('train_op_fn and optimizer cannot both be set.')
        train_op = optimizer.minimize(
            regularized_training_loss,
            global_step=tf.compat.v1.train.get_global_step())
      elif train_op_fn is not None:
        train_op = train_op_fn(regularized_training_loss)
      else:
        raise ValueError('train_op_fn and optimizer cannot both be None.')
      train_op = _append_update_ops(train_op)
      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == tf.compat.v1.losses.Reduction.SUM:
        example_weight_sum = tf.math.reduce_sum(
            weights * tf.compat.v1.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None
    with tf.compat.v1.name_scope(''):
      keys = metric_keys.MetricKeys
      tf.compat.v1.summary.scalar(
          _summary_key(self._name, keys.LOSS), scalar_loss)
      if mean_loss is not None:
        tf.compat.v1.summary.scalar(
            _summary_key(self._name, keys.LOSS_MEAN), mean_loss)
      if regularization_loss is not None:
        tf.compat.v1.summary.scalar(
            _summary_key(self._name, keys.LOSS_REGULARIZATION),
            regularization_loss)
    return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
        mode=ModeKeys.TRAIN,
        predictions=predictions,
        loss=scalar_loss,
        train_op=train_op)


def _binary_logistic_or_multi_class_head(n_classes, weight_column,
                                         label_vocabulary, loss_reduction):
  """Creates either binary or multi-class head.

  Args:
    n_classes: Number of label classes.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `_NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
      weight_column.normalizer_fn is applied on it to get weight tensor.
    label_vocabulary: A list of strings represents possible label values. If
      given, labels must be string type and have any value in
      `label_vocabulary`. If it is not given, that means labels are already
      encoded as integer or float within [0, 1] for `n_classes=2` and encoded as
      integer values in {0, 1,..., n_classes-1} for `n_classes`>2 . Also there
      will be errors if vocabulary is not provided and labels are string.
    loss_reduction: Describes how to reduce training loss over batch.
      Defaults to `SUM`.

  Returns:
    `head._Head` instance.
  """
  if n_classes == 2:
    head = _binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
  else:
    head = _multi_class_head_with_softmax_cross_entropy_loss(
        n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
  return head
