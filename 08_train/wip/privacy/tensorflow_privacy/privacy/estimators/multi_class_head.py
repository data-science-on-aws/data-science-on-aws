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
"""Multiclass head for Estimator that allow integration with TF Privacy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


class DPMultiClassHead(tf.estimator.MultiClassHead):
  """Creates a TF Privacy-enabled version of MultiClassHead."""

  def __init__(self,
               n_classes,
               weight_column=None,
               label_vocabulary=None,
               loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               loss_fn=None,
               name=None):
    super(DPMultiClassHead, self).__init__(
        n_classes=n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        loss_fn=loss_fn,
        name=name)

  def loss(self,
           labels,
           logits,
           features=None,
           mode=None,
           regularization_losses=None):
    """Returns regularized training loss. See `base_head.Head` for details."""
    del mode  # Unused for this head.
    with tf.compat.v1.name_scope(
        'losses', values=(logits, labels, regularization_losses, features)):
      logits = base_head.check_logits_final_dim(logits, self.logits_dimension)
      labels = self._processed_labels(logits, labels)
      unweighted_loss, weights = self._unweighted_loss_and_weights(
          logits, labels, features)
      vector_training_loss = losses_utils.compute_weighted_loss(
          unweighted_loss,
          sample_weight=weights,
          reduction=tf.keras.losses.Reduction.NONE)
      regularization_loss = tf.math.add_n(
          regularization_losses) if regularization_losses is not None else None
      vector_regularized_training_loss = (
          tf.add(vector_training_loss, regularization_loss)
          if regularization_loss is not None else vector_training_loss)

    return vector_regularized_training_loss

  def _create_tpu_estimator_spec(self,
                                 features,
                                 mode,
                                 logits,
                                 labels=None,
                                 optimizer=None,
                                 trainable_variables=None,
                                 train_op_fn=None,
                                 update_ops=None,
                                 regularization_losses=None):
    """See superclass for description."""

    with tf.compat.v1.name_scope(self._name, 'head'):
      # Predict.
      pred_keys = prediction_keys.PredictionKeys
      predictions = self.predictions(logits)
      if mode == ModeKeys.PREDICT:
        probabilities = predictions[pred_keys.PROBABILITIES]
        classifier_output = base_head.classification_output(
            scores=probabilities,
            n_classes=self._n_classes,
            label_vocabulary=self._label_vocabulary)
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                base_head.DEFAULT_SERVING_KEY:
                    classifier_output,
                base_head.CLASSIFY_SERVING_KEY:
                    classifier_output,
                base_head.PREDICT_SERVING_KEY:
                    export_output.PredictOutput(predictions)
            })
      regularized_training_loss = self.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=mode,
          regularization_losses=regularization_losses)
      scalar_loss = tf.reduce_mean(regularized_training_loss)
      # Eval.
      if mode == ModeKeys.EVAL:
        eval_metrics = self.metrics(regularization_losses=regularization_losses)
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=ModeKeys.EVAL,
            predictions=predictions,
            loss=scalar_loss,
            eval_metrics=base_head.create_eval_metrics_tuple(
                self.update_metrics, {
                    'eval_metrics': eval_metrics,
                    'features': features,
                    'logits': logits,
                    'labels': labels,
                    'regularization_losses': regularization_losses
                }))
      # Train.
      train_op = base_head.create_estimator_spec_train_op(
          head_name=self._name,
          optimizer=optimizer,
          train_op_fn=train_op_fn,
          update_ops=update_ops,
          trainable_variables=trainable_variables,
          regularized_training_loss=regularized_training_loss,
          loss_reduction=self._loss_reduction)
    # Create summary.
    base_head.create_estimator_spec_summary(
        regularized_training_loss=scalar_loss,
        regularization_losses=regularization_losses,
        summary_key_fn=self._summary_key)
    return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
        mode=ModeKeys.TRAIN,
        predictions=predictions,
        loss=scalar_loss,
        train_op=train_op)
