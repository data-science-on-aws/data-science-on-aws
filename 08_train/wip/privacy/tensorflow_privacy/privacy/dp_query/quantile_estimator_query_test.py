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

"""Tests for QuantileAdaptiveClipSumQuery."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import quantile_estimator_query
from tensorflow_privacy.privacy.dp_query import test_utils

tf.enable_eager_execution()


def _make_quantile_estimator_query(
    initial_estimate,
    target_quantile,
    learning_rate,
    below_estimate_stddev,
    expected_num_records,
    geometric_update):
  if expected_num_records is not None:
    return quantile_estimator_query.QuantileEstimatorQuery(
        initial_estimate,
        target_quantile,
        learning_rate,
        below_estimate_stddev,
        expected_num_records,
        geometric_update)
  else:
    return quantile_estimator_query.NoPrivacyQuantileEstimatorQuery(
        initial_estimate,
        target_quantile,
        learning_rate,
        geometric_update)


class QuantileEstimatorQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('exact', True), ('fixed', False))
  def test_target_zero(self, exact):
    record1 = tf.constant(8.5)
    record2 = tf.constant(7.25)

    query = _make_quantile_estimator_query(
        initial_estimate=10.0,
        target_quantile=0.0,
        learning_rate=1.0,
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=False)

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 10.0)

    # On the first two iterations, both records are below, so the estimate goes
    # down by 1.0 (the learning rate). When the estimate reaches 8.0, only one
    # record is below, so the estimate goes down by only 0.5. After two more
    # iterations, both records are below, and the estimate stays there (at 7.0).

    expected_estimates = [9.0, 8.0, 7.5, 7.0, 7.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state)

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(('exact', True), ('fixed', False))
  def test_target_zero_geometric(self, exact):
    record1 = tf.constant(5.0)
    record2 = tf.constant(2.5)

    query = _make_quantile_estimator_query(
        initial_estimate=16.0,
        target_quantile=0.0,
        learning_rate=np.log(2.0),        # Geometric steps in powers of 2.
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=True)

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 16.0)

    # For two iterations, both records are below, so the estimate is halved.
    # Then only one record is below, so the estimate goes down by only sqrt(2.0)
    # to 4 / sqrt(2.0). Still only one record is below, so it reduces to 2.0.
    # Now no records are below, and the estimate norm stays there (at 2.0).

    four_div_root_two = 4 / np.sqrt(2.0)   # approx 2.828

    expected_estimates = [8.0, 4.0, four_div_root_two, 2.0, 2.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state)

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(('exact', True), ('fixed', False))
  def test_target_one(self, exact):
    record1 = tf.constant(1.5)
    record2 = tf.constant(2.75)

    query = _make_quantile_estimator_query(
        initial_estimate=0.0,
        target_quantile=1.0,
        learning_rate=1.0,
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=False)

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 0.0)

    # On the first two iterations, both are above, so the estimate goes up
    # by 1.0 (the learning rate). When it reaches 2.0, only one record is
    # above, so the estimate goes up by only 0.5. After two more iterations,
    # both records are below, and the estimate stays there (at 3.0).

    expected_estimates = [1.0, 2.0, 2.5, 3.0, 3.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state)

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(('exact', True), ('fixed', False))
  def test_target_one_geometric(self, exact):
    record1 = tf.constant(1.5)
    record2 = tf.constant(3.0)

    query = _make_quantile_estimator_query(
        initial_estimate=0.5,
        target_quantile=1.0,
        learning_rate=np.log(2.0),        # Geometric steps in powers of 2.
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=True)

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 0.5)

    # On the first two iterations, both are above, so the estimate is doubled.
    # When the estimate reaches 2.0, only one record is above, so the estimate
    # is multiplied by sqrt(2.0). Still only one is above so it increases to
    # 4.0. Now both records are above, and the estimate stays there (at 4.0).

    two_times_root_two = 2 * np.sqrt(2.0)   # approx 2.828

    expected_estimates = [1.0, 2.0, two_times_root_two, 4.0, 4.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state)

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(
      ('start_low_geometric_exact', True, True, True),
      ('start_low_arithmetic_exact', True, True, False),
      ('start_high_geometric_exact', True, False, True),
      ('start_high_arithmetic_exact', True, False, False),
      ('start_low_geometric_noised', False, True, True),
      ('start_low_arithmetic_noised', False, True, False),
      ('start_high_geometric_noised', False, False, True),
      ('start_high_arithmetic_noised', False, False, False))
  def test_linspace(self, exact, start_low, geometric):
    # 100 records equally spaced from 0 to 10 in 0.1 increments.
    # Test that we converge to the correct median value and bounce around it.
    num_records = 21
    records = [tf.constant(x) for x in np.linspace(
        0.0, 10.0, num=num_records, dtype=np.float32)]

    query = _make_quantile_estimator_query(
        initial_estimate=(1.0 if start_low else 10.0),
        target_quantile=0.5,
        learning_rate=1.0,
        below_estimate_stddev=(0.0 if exact else 1e-2),
        expected_num_records=(None if exact else num_records),
        geometric_update=geometric)

    global_state = query.initial_global_state()

    for t in range(50):
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_estimate = global_state.current_estimate

      if t > 40:
        self.assertNear(actual_estimate, 5.0, 0.25)

  @parameterized.named_parameters(
      ('start_low_geometric_exact', True, True, True),
      ('start_low_arithmetic_exact', True, True, False),
      ('start_high_geometric_exact', True, False, True),
      ('start_high_arithmetic_exact', True, False, False),
      ('start_low_geometric_noised', False, True, True),
      ('start_low_arithmetic_noised', False, True, False),
      ('start_high_geometric_noised', False, False, True),
      ('start_high_arithmetic_noised', False, False, False))
  def test_all_equal(self, exact, start_low, geometric):
    # 20 equal records. Test that we converge to that record and bounce around
    # it. Unlike the linspace test, the quantile-matching objective is very
    # sharp at the optimum so a decaying learning rate is necessary.
    num_records = 20
    records = [tf.constant(5.0)] * num_records

    learning_rate = tf.Variable(1.0)

    query = _make_quantile_estimator_query(
        initial_estimate=(1.0 if start_low else 10.0),
        target_quantile=0.5,
        learning_rate=learning_rate,
        below_estimate_stddev=(0.0 if exact else 1e-2),
        expected_num_records=(None if exact else num_records),
        geometric_update=geometric)

    global_state = query.initial_global_state()

    for t in range(50):
      tf.assign(learning_rate, 1.0 / np.sqrt(t + 1))
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_estimate = global_state.current_estimate

      if t > 40:
        self.assertNear(actual_estimate, 5.0, 0.5)

  def test_raises_with_non_scalar_record(self):
    query = quantile_estimator_query.NoPrivacyQuantileEstimatorQuery(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0)

    with self.assertRaisesRegex(ValueError, 'scalar'):
      query.accumulate_record(None, None, [1.0, 2.0])


if __name__ == '__main__':
  tf.test.main()
