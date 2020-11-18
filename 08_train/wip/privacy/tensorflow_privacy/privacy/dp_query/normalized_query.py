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

"""Implements DPQuery interface for normalized queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import dp_query


class NormalizedQuery(dp_query.SumAggregationDPQuery):
  """DPQuery for queries with a DPQuery numerator and fixed denominator."""

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['numerator_state', 'denominator'])

  def __init__(self, numerator_query, denominator):
    """Initializer for NormalizedQuery.

    Args:
      numerator_query: A SumAggregationDPQuery for the numerator.
      denominator: A value for the denominator. May be None if it will be
        supplied via the set_denominator function before get_noised_result is
        called.
    """
    self._numerator = numerator_query
    self._denominator = denominator

    assert isinstance(self._numerator, dp_query.SumAggregationDPQuery)

  def set_ledger(self, ledger):
    self._numerator.set_ledger(ledger)

  def initial_global_state(self):
    if self._denominator is not None:
      denominator = tf.cast(self._denominator, tf.float32)
    else:
      denominator = None
    return self._GlobalState(
        self._numerator.initial_global_state(), denominator)

  def derive_sample_params(self, global_state):
    return self._numerator.derive_sample_params(global_state.numerator_state)

  def initial_sample_state(self, template):
    # NormalizedQuery has no sample state beyond the numerator state.
    return self._numerator.initial_sample_state(template)

  def preprocess_record(self, params, record):
    return self._numerator.preprocess_record(params, record)

  def get_noised_result(self, sample_state, global_state):
    noised_sum, new_sum_global_state = self._numerator.get_noised_result(
        sample_state, global_state.numerator_state)
    def normalize(v):
      return tf.truediv(v, global_state.denominator)

    return (tf.nest.map_structure(normalize, noised_sum),
            self._GlobalState(new_sum_global_state, global_state.denominator))

  def derive_metrics(self, global_state):
    return self._numerator.derive_metrics(global_state.numerator_state)
