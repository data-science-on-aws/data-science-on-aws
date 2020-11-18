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

"""Implements DPQuery interface for quantile estimator.

From a starting estimate of the target quantile, the estimate is updated
dynamically where the fraction of below_estimate updates is estimated in a
differentially private manner. For details see Thakkar et al., "Differentially
Private Learning with Adaptive Clipping" [http://arxiv.org/abs/1905.03871].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import no_privacy_query


class QuantileEstimatorQuery(dp_query.SumAggregationDPQuery):
  """Iterative process to estimate target quantile of a univariate distribution.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', [
          'current_estimate',
          'target_quantile',
          'learning_rate',
          'below_estimate_state'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['current_estimate', 'below_estimate_params'])

  # No separate SampleState-- sample state is just below_estimate_query's
  # SampleState.

  def __init__(
      self,
      initial_estimate,
      target_quantile,
      learning_rate,
      below_estimate_stddev,
      expected_num_records,
      geometric_update=False):
    """Initializes the QuantileAdaptiveClipSumQuery.

    Args:
      initial_estimate: The initial estimate of the quantile.
      target_quantile: The target quantile. I.e., a value of 0.8 means a value
        should be found for which approximately 80% of updates are
        less than the estimate each round.
      learning_rate: The learning rate. A rate of r means that the estimate
        will change by a maximum of r at each step (for arithmetic updating) or
        by a maximum factor of exp(r) (for geometric updating).
      below_estimate_stddev: The stddev of the noise added to the count of
        records currently below the estimate. Since the sensitivity of the count
        query is 0.5, as a rule of thumb it should be about 0.5 for reasonable
        privacy.
      expected_num_records: The expected number of records per round.
      geometric_update: If True, use geometric updating of estimate. Geometric
        updating is preferred for non-negative records like vector norms that
        could potentially be very large or very close to zero.
    """
    self._initial_estimate = initial_estimate
    self._target_quantile = target_quantile
    self._learning_rate = learning_rate

    self._below_estimate_query = self._construct_below_estimate_query(
        below_estimate_stddev, expected_num_records)
    assert isinstance(self._below_estimate_query,
                      dp_query.SumAggregationDPQuery)

    self._geometric_update = geometric_update

  def _construct_below_estimate_query(
      self, below_estimate_stddev, expected_num_records):
    # A DPQuery used to estimate the fraction of records that are less than the
    # current quantile estimate. It accumulates an indicator 0/1 of whether each
    # record is below the estimate, and normalizes by the expected number of
    # records. In practice, we accumulate counts shifted by -0.5 so they are
    # centered at zero. This makes the sensitivity of the below_estimate count
    # query 0.5 instead of 1.0, since the maximum that a single record could
    # affect the count is 0.5. Note that although the l2_norm_clip of the
    # below_estimate query is 0.5, no clipping will ever actually occur
    # because the value of each record is always +/-0.5.
    return gaussian_query.GaussianAverageQuery(
        l2_norm_clip=0.5,
        sum_stddev=below_estimate_stddev,
        denominator=expected_num_records)

  def set_ledger(self, ledger):
    self._below_estimate_query.set_ledger(ledger)

  def initial_global_state(self):
    return self._GlobalState(
        tf.cast(self._initial_estimate, tf.float32),
        tf.cast(self._target_quantile, tf.float32),
        tf.cast(self._learning_rate, tf.float32),
        self._below_estimate_query.initial_global_state())

  def derive_sample_params(self, global_state):
    below_estimate_params = self._below_estimate_query.derive_sample_params(
        global_state.below_estimate_state)
    return self._SampleParams(global_state.current_estimate,
                              below_estimate_params)

  def initial_sample_state(self, template=None):
    # Template is ignored because records are required to be scalars.
    del template

    return self._below_estimate_query.initial_sample_state(0.0)

  def preprocess_record(self, params, record):
    tf.debugging.assert_scalar(record)

    # We accumulate counts shifted by 0.5 so they are centered at zero.
    # This makes the sensitivity of the count query 0.5 instead of 1.0.
    below = tf.cast(record <= params.current_estimate, tf.float32) - 0.5
    return self._below_estimate_query.preprocess_record(
        params.below_estimate_params, below)

  def get_noised_result(self, sample_state, global_state):
    below_estimate_result, new_below_estimate_state = (
        self._below_estimate_query.get_noised_result(
            sample_state,
            global_state.below_estimate_state))

    # Unshift below_estimate percentile by 0.5. (See comment in initializer.)
    below_estimate = below_estimate_result + 0.5

    # Protect against out-of-range estimates.
    below_estimate = tf.minimum(1.0, tf.maximum(0.0, below_estimate))

    # Loss function is convex, with derivative in [-1, 1], and minimized when
    # the true quantile matches the target.
    loss_grad = below_estimate - global_state.target_quantile

    update = global_state.learning_rate * loss_grad

    if self._geometric_update:
      new_estimate = global_state.current_estimate * tf.math.exp(-update)
    else:
      new_estimate = global_state.current_estimate - update

    new_global_state = global_state._replace(
        current_estimate=new_estimate,
        below_estimate_state=new_below_estimate_state)

    return new_estimate, new_global_state

  def derive_metrics(self, global_state):
    return collections.OrderedDict(estimate=global_state.current_estimate)


class NoPrivacyQuantileEstimatorQuery(QuantileEstimatorQuery):
  """Iterative process to estimate target quantile of a univariate distribution.

  Unlike the base class, this uses a NoPrivacyQuery to estimate the fraction
  below estimate with an exact denominator.
  """

  def __init__(
      self,
      initial_estimate,
      target_quantile,
      learning_rate,
      geometric_update=False):
    """Initializes the NoPrivacyQuantileEstimatorQuery.

    Args:
      initial_estimate: The initial estimate of the quantile.
      target_quantile: The target quantile. I.e., a value of 0.8 means a value
        should be found for which approximately 80% of updates are
        less than the estimate each round.
      learning_rate: The learning rate. A rate of r means that the estimate
        will change by a maximum of r at each step (for arithmetic updating) or
        by a maximum factor of exp(r) (for geometric updating).
      geometric_update: If True, use geometric updating of estimate. Geometric
        updating is preferred for non-negative records like vector norms that
        could potentially be very large or very close to zero.
    """
    super(NoPrivacyQuantileEstimatorQuery, self).__init__(
        initial_estimate, target_quantile, learning_rate,
        below_estimate_stddev=None, expected_num_records=None,
        geometric_update=geometric_update)

  def _construct_below_estimate_query(
      self, below_estimate_stddev, expected_num_records):
    del below_estimate_stddev
    del expected_num_records
    return no_privacy_query.NoPrivacyAverageQuery()
