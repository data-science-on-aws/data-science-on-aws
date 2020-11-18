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
"""A lightweight buffer for maintaining tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class TensorBuffer(object):
  """A lightweight buffer for maintaining lists.

  The TensorBuffer accumulates tensors of the given shape into a tensor (whose
  rank is one more than that of the given shape) via calls to `append`. The
  current value of the accumulated tensor can be extracted via the property
  `values`.
  """

  def __init__(self, capacity, shape, dtype=tf.int32, name=None):
    """Initializes the TensorBuffer.

    Args:
      capacity: Initial capacity. Buffer will double in capacity each time it is
        filled to capacity.
      shape: The shape (as tuple or list) of the tensors to accumulate.
      dtype: The type of the tensors.
      name: A string name for the variable_scope used.

    Raises:
      ValueError: If the shape is empty (specifies scalar shape).
    """
    shape = list(shape)
    self._rank = len(shape)
    self._name = name
    self._dtype = dtype
    if not self._rank:
      raise ValueError('Shape cannot be scalar.')
    shape = [capacity] + shape

    with tf.variable_scope(self._name):
      # We need to use a placeholder as the initial value to allow resizing.
      self._buffer = tf.Variable(
          initial_value=tf.placeholder_with_default(
              tf.zeros(shape, dtype), shape=None),
          trainable=False,
          name='buffer',
          use_resource=True)
      self._current_size = tf.Variable(
          initial_value=0, dtype=tf.int32, trainable=False, name='current_size')
      self._capacity = tf.Variable(
          initial_value=capacity,
          dtype=tf.int32,
          trainable=False,
          name='capacity')

  def append(self, value):
    """Appends a new tensor to the end of the buffer.

    Args:
      value: The tensor to append. Must match the shape specified in the
        initializer.

    Returns:
      An op appending the new tensor to the end of the buffer.
    """

    def _double_capacity():
      """Doubles the capacity of the current tensor buffer."""
      padding = tf.zeros_like(self._buffer, self._buffer.dtype)
      new_buffer = tf.concat([self._buffer, padding], axis=0)
      if tf.executing_eagerly():
        with tf.variable_scope(self._name, reuse=True):
          self._buffer = tf.get_variable(
              name='buffer',
              dtype=self._dtype,
              initializer=new_buffer,
              trainable=False)
          return self._buffer, tf.assign(
              self._capacity, tf.multiply(self._capacity, 2))
      else:
        return tf.assign(
            self._buffer, new_buffer,
            validate_shape=False), tf.assign(
                self._capacity, tf.multiply(self._capacity, 2))

    update_buffer, update_capacity = tf.cond(
        pred=tf.equal(self._current_size, self._capacity),
        true_fn=_double_capacity,
        false_fn=lambda: (self._buffer, self._capacity))

    with tf.control_dependencies([update_buffer, update_capacity]):
      with tf.control_dependencies([
          tf.assert_less(
              self._current_size,
              self._capacity,
              message='Appending past end of TensorBuffer.'),
          tf.assert_equal(
              tf.shape(input=value),
              tf.shape(input=self._buffer)[1:],
              message='Appending value of inconsistent shape.')
      ]):
        with tf.control_dependencies(
            [tf.assign(self._buffer[self._current_size, :], value)]):
          return tf.assign_add(self._current_size, 1)

  @property
  def values(self):
    """Returns the accumulated tensor."""
    begin_value = tf.zeros([self._rank + 1], dtype=tf.int32)
    value_size = tf.concat([[self._current_size],
                            tf.constant(-1, tf.int32, [self._rank])], 0)
    return tf.slice(self._buffer, begin_value, value_size)

  @property
  def current_size(self):
    """Returns the current number of tensors in the buffer."""
    return self._current_size

  @property
  def capacity(self):
    """Returns the current capacity of the buffer."""
    return self._capacity
