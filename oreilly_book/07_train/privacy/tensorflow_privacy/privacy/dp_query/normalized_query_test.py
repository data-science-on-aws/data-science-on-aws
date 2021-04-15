# Copyright 2019, The TensorFlow Authors.
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

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import normalized_query
from tensorflow_privacy.privacy.dp_query import test_utils


class NormalizedQueryTest(tf.test.TestCase):

  def test_normalization(self):
    with self.cached_session() as sess:
      record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
      record2 = tf.constant([4.0, -3.0])  # Not clipped.

      sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip=5.0, stddev=0.0)
      query = normalized_query.NormalizedQuery(
          numerator_query=sum_query, denominator=2.0)

      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [0.5, 0.5]
      self.assertAllClose(result, expected)


if __name__ == '__main__':
  tf.test.main()
