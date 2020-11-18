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
"""Tests for tensor_buffer in graph mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import tensor_buffer


class TensorBufferTest(tf.test.TestCase):
  """Tests for TensorBuffer in graph mode."""

  def test_noresize(self):
    """Test buffer does not resize if capacity is not exceeded."""
    with self.cached_session() as sess:
      size, shape = 2, [2, 3]

      my_buffer = tensor_buffer.TensorBuffer(size, shape, name='my_buffer')
      value1 = [[1, 2, 3], [4, 5, 6]]
      with tf.control_dependencies([my_buffer.append(value1)]):
        value2 = [[7, 8, 9], [10, 11, 12]]
        with tf.control_dependencies([my_buffer.append(value2)]):
          values = my_buffer.values
          current_size = my_buffer.current_size
          capacity = my_buffer.capacity
      self.evaluate(tf.global_variables_initializer())

      v, cs, cap = sess.run([values, current_size, capacity])
      self.assertAllEqual(v, [value1, value2])
      self.assertEqual(cs, 2)
      self.assertEqual(cap, 2)

  def test_resize(self):
    """Test buffer resizes if capacity is exceeded."""
    with self.cached_session() as sess:
      size, shape = 2, [2, 3]

      my_buffer = tensor_buffer.TensorBuffer(size, shape, name='my_buffer')
      value1 = [[1, 2, 3], [4, 5, 6]]
      with tf.control_dependencies([my_buffer.append(value1)]):
        value2 = [[7, 8, 9], [10, 11, 12]]
        with tf.control_dependencies([my_buffer.append(value2)]):
          value3 = [[13, 14, 15], [16, 17, 18]]
          with tf.control_dependencies([my_buffer.append(value3)]):
            values = my_buffer.values
            current_size = my_buffer.current_size
            capacity = my_buffer.capacity
      self.evaluate(tf.global_variables_initializer())

      v, cs, cap = sess.run([values, current_size, capacity])
      self.assertAllEqual(v, [value1, value2, value3])
      self.assertEqual(cs, 3)
      self.assertEqual(cap, 4)


if __name__ == '__main__':
  tf.test.main()
