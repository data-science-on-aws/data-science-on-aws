# Copyright 2018, The TensorFlow Authors.
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

"""Tests for NestedQuery."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import nested_query
from tensorflow_privacy.privacy.dp_query import test_utils


_basic_query = gaussian_query.GaussianSumQuery(1.0, 0.0)


class NestedQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_nested_gaussian_sum_no_clip_no_noise(self):
    with self.cached_session() as sess:
      query1 = gaussian_query.GaussianSumQuery(
          l2_norm_clip=10.0, stddev=0.0)
      query2 = gaussian_query.GaussianSumQuery(
          l2_norm_clip=10.0, stddev=0.0)

      query = nested_query.NestedSumQuery([query1, query2])

      record1 = [1.0, [2.0, 3.0]]
      record2 = [4.0, [3.0, 2.0]]

      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [5.0, [5.0, 5.0]]
      self.assertAllClose(result, expected)

  def test_nested_gaussian_average_no_clip_no_noise(self):
    with self.cached_session() as sess:
      query1 = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=10.0, sum_stddev=0.0, denominator=5.0)
      query2 = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=10.0, sum_stddev=0.0, denominator=5.0)

      query = nested_query.NestedSumQuery([query1, query2])

      record1 = [1.0, [2.0, 3.0]]
      record2 = [4.0, [3.0, 2.0]]

      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1.0, [1.0, 1.0]]
      self.assertAllClose(result, expected)

  def test_nested_gaussian_average_with_clip_no_noise(self):
    with self.cached_session() as sess:
      query1 = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=4.0, sum_stddev=0.0, denominator=5.0)
      query2 = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=5.0, sum_stddev=0.0, denominator=5.0)

      query = nested_query.NestedSumQuery([query1, query2])

      record1 = [1.0, [12.0, 9.0]]  # Clipped to [1.0, [4.0, 3.0]]
      record2 = [5.0, [1.0, 2.0]]   # Clipped to [4.0, [1.0, 2.0]]

      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1.0, [1.0, 1.0]]
      self.assertAllClose(result, expected)

  def test_complex_nested_query(self):
    with self.cached_session() as sess:
      query_ab = gaussian_query.GaussianSumQuery(
          l2_norm_clip=1.0, stddev=0.0)
      query_c = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=10.0, sum_stddev=0.0, denominator=2.0)
      query_d = gaussian_query.GaussianSumQuery(
          l2_norm_clip=10.0, stddev=0.0)

      query = nested_query.NestedSumQuery(
          [query_ab, {'c': query_c, 'd': [query_d]}])

      record1 = [{'a': 0.0, 'b': 2.71828}, {'c': (-4.0, 6.0), 'd': [-4.0]}]
      record2 = [{'a': 3.14159, 'b': 0.0}, {'c': (6.0, -4.0), 'd': [5.0]}]

      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [{'a': 1.0, 'b': 1.0}, {'c': (1.0, 1.0), 'd': [1.0]}]
      self.assertAllClose(result, expected)

  def test_nested_query_with_noise(self):
    with self.cached_session() as sess:
      sum_stddev = 2.71828
      denominator = 3.14159

      query1 = gaussian_query.GaussianSumQuery(
          l2_norm_clip=1.5, stddev=sum_stddev)
      query2 = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=0.5, sum_stddev=sum_stddev, denominator=denominator)
      query = nested_query.NestedSumQuery((query1, query2))

      record1 = (3.0, [2.0, 1.5])
      record2 = (0.0, [-1.0, -3.5])

      query_result, _ = test_utils.run_query(query, [record1, record2])

      noised_averages = []
      for _ in range(1000):
        noised_averages.append(tf.nest.flatten(sess.run(query_result)))

      result_stddev = np.std(noised_averages, 0)
      avg_stddev = sum_stddev / denominator
      expected_stddev = [sum_stddev, avg_stddev, avg_stddev]
      self.assertArrayNear(result_stddev, expected_stddev, 0.1)

  @parameterized.named_parameters(
      ('type_mismatch', [_basic_query], (1.0,), TypeError),
      ('too_many_queries', [_basic_query, _basic_query], [1.0], ValueError),
      ('query_too_deep', [_basic_query, [_basic_query]], [1.0, 1.0], TypeError))
  def test_record_incompatible_with_query(
      self, queries, record, error_type):
    with self.assertRaises(error_type):
      test_utils.run_query(nested_query.NestedSumQuery(queries), [record])

  def test_raises_with_non_sum(self):
    class NonSumDPQuery(dp_query.DPQuery):
      pass

    non_sum_query = NonSumDPQuery()

    # This should work.
    nested_query.NestedQuery(non_sum_query)

    # This should not.
    with self.assertRaises(TypeError):
      nested_query.NestedSumQuery(non_sum_query)

  def test_metrics(self):
    class QueryWithMetric(dp_query.SumAggregationDPQuery):

      def __init__(self, metric_val):
        self._metric_val = metric_val

      def derive_metrics(self, global_state):
        return collections.OrderedDict(metric=self._metric_val)

    query1 = QueryWithMetric(1)
    query2 = QueryWithMetric(2)
    query3 = QueryWithMetric(3)

    nested_a = nested_query.NestedSumQuery(query1)
    global_state = nested_a.initial_global_state()
    metric_val = nested_a.derive_metrics(global_state)
    self.assertEqual(metric_val['metric'], 1)

    nested_b = nested_query.NestedSumQuery(
        {'key1': query1, 'key2': [query2, query3]})
    global_state = nested_b.initial_global_state()
    metric_val = nested_b.derive_metrics(global_state)
    self.assertEqual(metric_val['key1/metric'], 1)
    self.assertEqual(metric_val['key2/0/metric'], 2)
    self.assertEqual(metric_val['key2/1/metric'], 3)


if __name__ == '__main__':
  tf.test.main()
