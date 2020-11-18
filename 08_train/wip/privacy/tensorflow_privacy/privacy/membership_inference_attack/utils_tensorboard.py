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
"""Utility functions for writing attack results to tensorboard."""
from typing import List
from typing import Union

import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_flattened_attack_metrics


def write_to_tensorboard(writers, tags, values, step):
  """Write metrics to tensorboard.

  Args:
    writers: a list of tensorboard writers or one writer to be used for metrics.
    If it's a list, it should be of the same length as tags
    tags: a list of tags of metrics
    values: a list of values of metrics with the same length as tags
    step: step for the tensorboard summary
  """
  if writers is None or not writers:
    raise ValueError('write_to_tensorboard does not get any writer.')

  if not isinstance(writers, list):
    writers = [writers] * len(tags)

  assert len(writers) == len(tags) == len(values)

  for writer, tag, val in zip(writers, tags, values):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=val)
    writer.add_summary(summary, step)

  for writer in set(writers):
    writer.flush()


def write_results_to_tensorboard(
    attack_results: AttackResults,
    writers: Union[tf.summary.FileWriter, List[tf.summary.FileWriter]],
    step: int,
    merge_classifiers: bool):
  """Write attack results to tensorboard.

  Args:
    attack_results: results from attack
    writers: a list of tensorboard writers or one writer to be used for metrics
    step: step for the tensorboard summary
    merge_classifiers: if true, plot different classifiers with the same
      slicing_spec and metric in the same figure
  """
  if writers is None or not writers:
    raise ValueError('write_results_to_tensorboard does not get any writer.')

  att_types, att_slices, att_metrics, att_values = get_flattened_attack_metrics(
      attack_results)
  if merge_classifiers:
    att_tags = ['attack/' + '_'.join([s, m]) for s, m in
                zip(att_slices, att_metrics)]
    write_to_tensorboard([writers[t] for t in att_types],
                         att_tags, att_values, step)
  else:
    att_tags = ['attack/' + '_'.join([s, t, m]) for t, s, m in
                zip(att_types, att_slices, att_metrics)]
    write_to_tensorboard(writers, att_tags, att_values, step)
