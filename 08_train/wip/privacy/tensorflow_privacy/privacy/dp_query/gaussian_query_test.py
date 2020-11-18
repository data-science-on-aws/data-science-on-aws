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

"""Tests for GaussianAverageQuery."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import test_utils


class GaussianQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_gaussian_sum_no_clip_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([2.0, 0.0])
      record2 = tf.constant([-1.0, 1.0])

      query = gaussian_query.GaussianSumQuery(
          l2_norm_clip=10.0, stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1.0, 1.0]
      self.assertAllClose(result, expected)

  def test_gaussian_sum_with_clip_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
      record2 = tf.constant([4.0, -3.0])  # Not clipped.

      query = gaussian_query.GaussianSumQuery(
          l2_norm_clip=5.0, stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1.0, 1.0]
      self.assertAllClose(result, expected)

  def test_gaussian_sum_with_changing_clip_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
      record2 = tf.constant([4.0, -3.0])  # Not clipped.

      l2_norm_clip = tf.Variable(5.0)
      l2_norm_clip_placeholder = tf.placeholder(tf.float32)
      assign_l2_norm_clip = tf.assign(l2_norm_clip, l2_norm_clip_placeholder)
      query = gaussian_query.GaussianSumQuery(
          l2_norm_clip=l2_norm_clip, stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])

      self.evaluate(tf.global_variables_initializer())
      result = sess.run(query_result)
      expected = [1.0, 1.0]
      self.assertAllClose(result, expected)

      sess.run(assign_l2_norm_clip, {l2_norm_clip_placeholder: 0.0})
      result = sess.run(query_result)
      expected = [0.0, 0.0]
      self.assertAllClose(result, expected)

  def test_gaussian_sum_with_noise(self):
    with self.cached_session() as sess:
      record1, record2 = 2.71828, 3.14159
      stddev = 1.0

      query = gaussian_query.GaussianSumQuery(
          l2_norm_clip=5.0, stddev=stddev)
      query_result, _ = test_utils.run_query(query, [record1, record2])

      noised_sums = []
      for _ in xrange(1000):
        noised_sums.append(sess.run(query_result))

      result_stddev = np.std(noised_sums)
      self.assertNear(result_stddev, stddev, 0.1)

  def test_gaussian_sum_merge(self):
    records1 = [tf.constant([2.0, 0.0]), tf.constant([-1.0, 1.0])]
    records2 = [tf.constant([3.0, 5.0]), tf.constant([-1.0, 4.0])]

    def get_sample_state(records):
      query = gaussian_query.GaussianSumQuery(l2_norm_clip=10.0, stddev=1.0)
      global_state = query.initial_global_state()
      params = query.derive_sample_params(global_state)
      sample_state = query.initial_sample_state(records[0])
      for record in records:
        sample_state = query.accumulate_record(params, sample_state, record)
      return sample_state

    sample_state_1 = get_sample_state(records1)
    sample_state_2 = get_sample_state(records2)

    merged = gaussian_query.GaussianSumQuery(10.0, 1.0).merge_sample_states(
        sample_state_1,
        sample_state_2)

    with self.cached_session() as sess:
      result = sess.run(merged)

    expected = [3.0, 10.0]
    self.assertAllClose(result, expected)

  def test_gaussian_average_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([5.0, 0.0])   # Clipped to [3.0, 0.0].
      record2 = tf.constant([-1.0, 2.0])  # Not clipped.

      query = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=3.0, sum_stddev=0.0, denominator=2.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected_average = [1.0, 1.0]
      self.assertAllClose(result, expected_average)

  def test_gaussian_average_with_noise(self):
    with self.cached_session() as sess:
      record1, record2 = 2.71828, 3.14159
      sum_stddev = 1.0
      denominator = 2.0

      query = gaussian_query.GaussianAverageQuery(
          l2_norm_clip=5.0, sum_stddev=sum_stddev, denominator=denominator)
      query_result, _ = test_utils.run_query(query, [record1, record2])

      noised_averages = []
      for _ in range(1000):
        noised_averages.append(sess.run(query_result))

      result_stddev = np.std(noised_averages)
      avg_stddev = sum_stddev / denominator
      self.assertNear(result_stddev, avg_stddev, 0.1)

  @parameterized.named_parameters(
      ('type_mismatch', [1.0], (1.0,), TypeError),
      ('too_few_on_left', [1.0], [1.0, 1.0], ValueError),
      ('too_few_on_right', [1.0, 1.0], [1.0], ValueError))
  def test_incompatible_records(self, record1, record2, error_type):
    query = gaussian_query.GaussianSumQuery(1.0, 0.0)
    with self.assertRaises(error_type):
      test_utils.run_query(query, [record1, record2])


if __name__ == '__main__':
  tf.test.main()
