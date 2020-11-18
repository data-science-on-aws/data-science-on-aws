# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PrivacyLedger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import nested_query
from tensorflow_privacy.privacy.dp_query import test_utils

tf.enable_eager_execution()


class PrivacyLedgerTest(tf.test.TestCase):

  def test_fail_on_probability_zero(self):
    with self.assertRaisesRegexp(ValueError,
                                 'Selection probability cannot be 0.'):
      privacy_ledger.PrivacyLedger(10, 0)

  def test_basic(self):
    ledger = privacy_ledger.PrivacyLedger(10, 0.1)
    ledger.record_sum_query(5.0, 1.0)
    ledger.record_sum_query(2.0, 0.5)

    ledger.finalize_sample()

    expected_queries = [[5.0, 1.0], [2.0, 0.5]]
    formatted = ledger.get_formatted_ledger_eager()

    sample = formatted[0]
    self.assertAllClose(sample.population_size, 10.0)
    self.assertAllClose(sample.selection_probability, 0.1)
    self.assertAllClose(sorted(sample.queries), sorted(expected_queries))

  def test_sum_query(self):
    record1 = tf.constant([2.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    population_size = tf.Variable(0)
    selection_probability = tf.Variable(1.0)

    query = gaussian_query.GaussianSumQuery(
        l2_norm_clip=10.0, stddev=0.0)
    query = privacy_ledger.QueryWithLedger(
        query, population_size, selection_probability)

    # First sample.
    tf.assign(population_size, 10)
    tf.assign(selection_probability, 0.1)
    test_utils.run_query(query, [record1, record2])

    expected_queries = [[10.0, 0.0]]
    formatted = query.ledger.get_formatted_ledger_eager()
    sample_1 = formatted[0]
    self.assertAllClose(sample_1.population_size, 10.0)
    self.assertAllClose(sample_1.selection_probability, 0.1)
    self.assertAllClose(sample_1.queries, expected_queries)

    # Second sample.
    tf.assign(population_size, 20)
    tf.assign(selection_probability, 0.2)
    test_utils.run_query(query, [record1, record2])

    formatted = query.ledger.get_formatted_ledger_eager()
    sample_1, sample_2 = formatted
    self.assertAllClose(sample_1.population_size, 10.0)
    self.assertAllClose(sample_1.selection_probability, 0.1)
    self.assertAllClose(sample_1.queries, expected_queries)

    self.assertAllClose(sample_2.population_size, 20.0)
    self.assertAllClose(sample_2.selection_probability, 0.2)
    self.assertAllClose(sample_2.queries, expected_queries)

  def test_nested_query(self):
    population_size = tf.Variable(0)
    selection_probability = tf.Variable(1.0)

    query1 = gaussian_query.GaussianAverageQuery(
        l2_norm_clip=4.0, sum_stddev=2.0, denominator=5.0)
    query2 = gaussian_query.GaussianAverageQuery(
        l2_norm_clip=5.0, sum_stddev=1.0, denominator=5.0)

    query = nested_query.NestedQuery([query1, query2])
    query = privacy_ledger.QueryWithLedger(
        query, population_size, selection_probability)

    record1 = [1.0, [12.0, 9.0]]
    record2 = [5.0, [1.0, 2.0]]

    # First sample.
    tf.assign(population_size, 10)
    tf.assign(selection_probability, 0.1)
    test_utils.run_query(query, [record1, record2])

    expected_queries = [[4.0, 2.0], [5.0, 1.0]]
    formatted = query.ledger.get_formatted_ledger_eager()
    sample_1 = formatted[0]
    self.assertAllClose(sample_1.population_size, 10.0)
    self.assertAllClose(sample_1.selection_probability, 0.1)
    self.assertAllClose(sorted(sample_1.queries), sorted(expected_queries))

    # Second sample.
    tf.assign(population_size, 20)
    tf.assign(selection_probability, 0.2)
    test_utils.run_query(query, [record1, record2])

    formatted = query.ledger.get_formatted_ledger_eager()
    sample_1, sample_2 = formatted
    self.assertAllClose(sample_1.population_size, 10.0)
    self.assertAllClose(sample_1.selection_probability, 0.1)
    self.assertAllClose(sorted(sample_1.queries), sorted(expected_queries))

    self.assertAllClose(sample_2.population_size, 20.0)
    self.assertAllClose(sample_2.selection_probability, 0.2)
    self.assertAllClose(sorted(sample_2.queries), sorted(expected_queries))


if __name__ == '__main__':
  tf.test.main()
