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
"""Tests for tensor_buffer in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import tensor_buffer

tf.enable_eager_execution()


class TensorBufferTest(tf.test.TestCase):
  """Tests for TensorBuffer in eager mode."""

  def test_basic(self):
    size, shape = 2, [2, 3]

    my_buffer = tensor_buffer.TensorBuffer(size, shape, name='my_buffer')

    value1 = [[1, 2, 3], [4, 5, 6]]
    my_buffer.append(value1)
    self.assertAllEqual(my_buffer.values.numpy(), [value1])

    value2 = [[4, 5, 6], [7, 8, 9]]
    my_buffer.append(value2)
    self.assertAllEqual(my_buffer.values.numpy(), [value1, value2])

  def test_fail_on_scalar(self):
    with self.assertRaisesRegexp(ValueError, 'Shape cannot be scalar.'):
      tensor_buffer.TensorBuffer(1, ())

  def test_fail_on_inconsistent_shape(self):
    size, shape = 1, [2, 3]

    my_buffer = tensor_buffer.TensorBuffer(size, shape, name='my_buffer')

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'Appending value of inconsistent shape.'):
      my_buffer.append(tf.ones(shape=[3, 4], dtype=tf.int32))

  def test_resize(self):
    size, shape = 2, [2, 3]

    my_buffer = tensor_buffer.TensorBuffer(size, shape, name='my_buffer')

    # Append three buffers. Third one should succeed after resizing.
    value1 = [[1, 2, 3], [4, 5, 6]]
    my_buffer.append(value1)
    self.assertAllEqual(my_buffer.values.numpy(), [value1])
    self.assertAllEqual(my_buffer.current_size.numpy(), 1)
    self.assertAllEqual(my_buffer.capacity.numpy(), 2)

    value2 = [[4, 5, 6], [7, 8, 9]]
    my_buffer.append(value2)
    self.assertAllEqual(my_buffer.values.numpy(), [value1, value2])
    self.assertAllEqual(my_buffer.current_size.numpy(), 2)
    self.assertAllEqual(my_buffer.capacity.numpy(), 2)

    value3 = [[7, 8, 9], [10, 11, 12]]
    my_buffer.append(value3)
    self.assertAllEqual(my_buffer.values.numpy(), [value1, value2, value3])
    self.assertAllEqual(my_buffer.current_size.numpy(), 3)
    # Capacity should have doubled.
    self.assertAllEqual(my_buffer.capacity.numpy(), 4)


if __name__ == '__main__':
  tf.test.main()
