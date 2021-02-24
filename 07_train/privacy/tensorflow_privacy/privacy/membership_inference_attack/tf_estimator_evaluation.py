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
"""A hook and a function in tf estimator for membership inference attack."""

import os
from typing import Iterable
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_flattened_attack_metrics
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.utils import log_loss
from tensorflow_privacy.privacy.membership_inference_attack.utils_tensorboard import write_results_to_tensorboard


def calculate_losses(estimator, input_fn, labels):
  """Get predictions and losses for samples.

  The assumptions are 1) the loss is cross-entropy loss, and 2) user have
  specified prediction mode to return predictions, e.g.,
  when mode == tf.estimator.ModeKeys.PREDICT, the model function returns
  tf.estimator.EstimatorSpec(mode=mode, predictions=tf.nn.softmax(logits)).

  Args:
    estimator: model to make prediction
    input_fn: input function to be used in estimator.predict
    labels: array of size (n_samples, ), true labels of samples (integer valued)

  Returns:
    preds: probability vector of each sample
    loss: cross entropy loss of each sample
  """
  pred = np.array(list(estimator.predict(input_fn=input_fn)))
  loss = log_loss(labels, pred)
  return pred, loss


class MembershipInferenceTrainingHook(tf.estimator.SessionRunHook):
  """Training hook to perform membership inference attack on epoch end."""

  def __init__(
      self,
      estimator,
      in_train, out_train,
      input_fn_constructor,
      slicing_spec: SlicingSpec = None,
      attack_types: Iterable[AttackType] = (AttackType.THRESHOLD_ATTACK,),
      tensorboard_dir=None,
      tensorboard_merge_classifiers=False):
    """Initialize the hook.

    Args:
      estimator: model to be tested
      in_train: (in_training samples, in_training labels)
      out_train: (out_training samples, out_training labels)
      input_fn_constructor: a function that receives sample, label and construct
        the input_fn for model prediction
      slicing_spec: slicing specification of the attack
      attack_types: a list of attacks, each of type AttackType
      tensorboard_dir: directory for tensorboard summary
      tensorboard_merge_classifiers: if true, plot different classifiers with
      the same slicing_spec and metric in the same figure
    """
    in_train_data, self._in_train_labels = in_train
    out_train_data, self._out_train_labels = out_train

    # Define the input functions for both in and out-training samples.
    self._in_train_input_fn = input_fn_constructor(in_train_data,
                                                   self._in_train_labels)
    self._out_train_input_fn = input_fn_constructor(out_train_data,
                                                    self._out_train_labels)
    self._estimator = estimator
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

  def end(self, session):
    results = run_attack_helper(self._estimator,
                                self._in_train_input_fn,
                                self._out_train_input_fn,
                                self._in_train_labels, self._out_train_labels,
                                self._slicing_spec,
                                self._attack_types)
    logging.info(results)

    att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
        results)
    print('Attack result:')
    print('\n'.join(['  %s: %.4f' % (', '.join([s, t, m]), v) for t, s, m, v in
                     zip(att_types, att_slices, att_metrics, att_values)]))

    # Write to tensorboard if tensorboard_dir is specified
    global_step = self._estimator.get_variable_value('global_step')
    if self._writers is not None:
      write_results_to_tensorboard(results, self._writers, global_step,
                                   self._tensorboard_merge_classifiers)


def run_attack_on_tf_estimator_model(
    estimator, in_train, out_train,
    input_fn_constructor,
    slicing_spec: SlicingSpec = None,
    attack_types: Iterable[AttackType] = (AttackType.THRESHOLD_ATTACK,)):
  """Performs the attack in the end of training.

  Args:
    estimator: model to be tested
    in_train: (in_training samples, in_training labels)
    out_train: (out_training samples, out_training labels)
    input_fn_constructor: a function that receives sample, label and construct
      the input_fn for model prediction
    slicing_spec: slicing specification of the attack
    attack_types: a list of attacks, each of type AttackType
  Returns:
    Results of the attack
  """
  in_train_data, in_train_labels = in_train
  out_train_data, out_train_labels = out_train

  # Define the input functions for both in and out-training samples.
  in_train_input_fn = input_fn_constructor(in_train_data, in_train_labels)
  out_train_input_fn = input_fn_constructor(out_train_data, out_train_labels)

  # Call the helper to run the attack.
  results = run_attack_helper(estimator,
                              in_train_input_fn, out_train_input_fn,
                              in_train_labels, out_train_labels,
                              slicing_spec,
                              attack_types)
  logging.info('End of training attack:')
  logging.info(results)
  return results


def run_attack_helper(
    estimator,
    in_train_input_fn, out_train_input_fn,
    in_train_labels, out_train_labels,
    slicing_spec: SlicingSpec = None,
    attack_types: Iterable[AttackType] = (AttackType.THRESHOLD_ATTACK,)):
  """A helper function to perform attack.

  Args:
    estimator: model to be tested
    in_train_input_fn: input_fn for in training data
    out_train_input_fn: input_fn for out of training data
    in_train_labels: in training labels
    out_train_labels: out of training labels
    slicing_spec: slicing specification of the attack
    attack_types: a list of attacks, each of type AttackType
  Returns:
    Results of the attack
  """
  # Compute predictions and losses
  in_train_pred, in_train_loss = calculate_losses(estimator,
                                                  in_train_input_fn,
                                                  in_train_labels)
  out_train_pred, out_train_loss = calculate_losses(estimator,
                                                    out_train_input_fn,
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
