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
"""A callback and a function in keras for membership inference attack."""

import os
from typing import Iterable
from absl import logging

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_flattened_attack_metrics
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.utils import log_loss
from tensorflow_privacy.privacy.membership_inference_attack.utils_tensorboard import write_results_to_tensorboard


def calculate_losses(model, data, labels):
  """Calculate losses of model prediction on data, provided true labels.

  Args:
    model: model to make prediction
    data: samples
    labels: true labels of samples (integer valued)

  Returns:
    preds: probability vector of each sample
    loss: cross entropy loss of each sample
  """
  pred = model.predict(data)
  loss = log_loss(labels, pred)
  return pred, loss


class MembershipInferenceCallback(tf.keras.callbacks.Callback):
  """Callback to perform membership inference attack on epoch end."""

  def __init__(
      self,
      in_train, out_train,
      slicing_spec: SlicingSpec = None,
      attack_types: Iterable[AttackType] = (AttackType.THRESHOLD_ATTACK,),
      tensorboard_dir=None,
      tensorboard_merge_classifiers=False):
    """Initalizes the callback.

    Args:
      in_train: (in_training samples, in_training labels)
      out_train: (out_training samples, out_training labels)
      slicing_spec: slicing specification of the attack
      attack_types: a list of attacks, each of type AttackType
      tensorboard_dir: directory for tensorboard summary
      tensorboard_merge_classifiers: if true, plot different classifiers with
      the same slicing_spec and metric in the same figure
    """
    self._in_train_data, self._in_train_labels = in_train
    self._out_train_data, self._out_train_labels = out_train
    self._slicing_spec = slicing_spec
    self._attack_types = attack_types
    self._tensorboard_merge_classifiers = tensorboard_merge_classifiers
    if tensorboard_dir:
      if tensorboard_merge_classifiers:
        self._writers = {}
        with tf.Graph().as_default():
          for attack_type in attack_types:
            self._writers[attack_type.name] = tf.summary.FileWriter(
                os.path.join(tensorboard_dir, 'MI', attack_type.name))
      else:
        with tf.Graph().as_default():
          self._writers = tf.summary.FileWriter(
              os.path.join(tensorboard_dir, 'MI'))
      logging.info('Will write to tensorboard.')
    else:
      self._writers = None

  def on_epoch_end(self, epoch, logs=None):
    results = run_attack_on_keras_model(
        self.model,
        (self._in_train_data, self._in_train_labels),
        (self._out_train_data, self._out_train_labels),
        self._slicing_spec,
        self._attack_types)
    logging.info(results)

    att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
        results)
    print('Attack result:')
    print('\n'.join(['  %s: %.4f' % (', '.join([s, t, m]), v) for t, s, m, v in
                     zip(att_types, att_slices, att_metrics, att_values)]))

    # Write to tensorboard if tensorboard_dir is specified
    if self._writers is not None:
      write_results_to_tensorboard(results, self._writers, epoch,
                                   self._tensorboard_merge_classifiers)


def run_attack_on_keras_model(
    model, in_train, out_train,
    slicing_spec: SlicingSpec = None,
    attack_types: Iterable[AttackType] = (AttackType.THRESHOLD_ATTACK,)):
  """Performs the attack on a trained model.

  Args:
    model: model to be tested
    in_train: a (in_training samples, in_training labels) tuple
    out_train: a (out_training samples, out_training labels) tuple
    slicing_spec: slicing specification of the attack
    attack_types: a list of attacks, each of type AttackType
  Returns:
    Results of the attack
  """
  in_train_data, in_train_labels = in_train
  out_train_data, out_train_labels = out_train

  # Compute predictions and losses
  in_train_pred, in_train_loss = calculate_losses(model, in_train_data,
                                                  in_train_labels)
  out_train_pred, out_train_loss = calculate_losses(model, out_train_data,
                                                    out_train_labels)
  attack_input = AttackInputData(
      logits_train=in_train_pred, logits_test=out_train_pred,
      labels_train=in_train_labels, labels_test=out_train_labels,
      loss_train=in_train_loss, loss_test=out_train_loss
  )
  results = mia.run_attacks(attack_input,
                            slicing_spec=slicing_spec,
                            attack_types=attack_types)
  return results
