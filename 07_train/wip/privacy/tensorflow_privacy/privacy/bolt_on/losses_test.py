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
"""Unit testing for losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager  # pylint: disable=g-importing-member
from io import StringIO  # pylint: disable=g-importing-member
import sys
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.framework import test_util
from tensorflow.compat.v1.python.keras import keras_parameterized
from tensorflow.compat.v1.python.keras.regularizers import L1L2
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexBinaryCrossentropy
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexHuber
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexMixin


@contextmanager
def captured_output():
  """Capture std_out and std_err within context."""
  new_out, new_err = StringIO(), StringIO()
  old_out, old_err = sys.stdout, sys.stderr
  try:
    sys.stdout, sys.stderr = new_out, new_err
    yield sys.stdout, sys.stderr
  finally:
    sys.stdout, sys.stderr = old_out, old_err


class StrongConvexMixinTests(keras_parameterized.TestCase):
  """Tests for the StrongConvexMixin."""
  @parameterized.named_parameters([
      {'testcase_name': 'beta not implemented',
       'fn': 'beta',
       'args': [1]},
      {'testcase_name': 'gamma not implemented',
       'fn': 'gamma',
       'args': []},
      {'testcase_name': 'lipchitz not implemented',
       'fn': 'lipchitz_constant',
       'args': [1]},
      {'testcase_name': 'radius not implemented',
       'fn': 'radius',
       'args': []},
  ])

  def test_not_implemented(self, fn, args):
    """Test that the given fn's are not implemented on the mixin.

    Args:
      fn: fn on Mixin to test
      args: arguments to fn of Mixin
    """
    with self.assertRaises(NotImplementedError):
      loss = StrongConvexMixin()
      getattr(loss, fn, None)(*args)

  @parameterized.named_parameters([
      {'testcase_name': 'radius not implemented',
       'fn': 'kernel_regularizer',
       'args': []},
  ])
  def test_return_none(self, fn, args):
    """Test that fn of Mixin returns None.

    Args:
      fn: fn of Mixin to test
      args: arguments to fn of Mixin
    """
    loss = StrongConvexMixin()
    ret = getattr(loss, fn, None)(*args)
    self.assertEqual(ret, None)


class BinaryCrossesntropyTests(keras_parameterized.TestCase):
  """tests for BinaryCrossesntropy StrongConvex loss."""

  @parameterized.named_parameters([
      {'testcase_name': 'normal',
       'reg_lambda': 1,
       'C': 1,
       'radius_constant': 1
      },  # pylint: disable=invalid-name
  ])
  def test_init_params(self, reg_lambda, C, radius_constant):
    """Test initialization for given arguments.

    Args:
      reg_lambda: initialization value for reg_lambda arg
      C: initialization value for C arg
      radius_constant: initialization value for radius_constant arg
    """
    # test valid domains for each variable
    loss = StrongConvexBinaryCrossentropy(reg_lambda, C, radius_constant)
    self.assertIsInstance(loss, StrongConvexBinaryCrossentropy)

  @parameterized.named_parameters([
      {'testcase_name': 'negative c',
       'reg_lambda': 1,
       'C': -1,
       'radius_constant': 1
      },
      {'testcase_name': 'negative radius',
       'reg_lambda': 1,
       'C': 1,
       'radius_constant': -1
      },
      {'testcase_name': 'negative lambda',
       'reg_lambda': -1,
       'C': 1,
       'radius_constant': 1
      },  # pylint: disable=invalid-name
  ])
  def test_bad_init_params(self, reg_lambda, C, radius_constant):
    """Test invalid domain for given params. Should return ValueError.

    Args:
      reg_lambda: initialization value for reg_lambda arg
      C: initialization value for C arg
      radius_constant: initialization value for radius_constant arg
    """
    # test valid domains for each variable
    with self.assertRaises(ValueError):
      StrongConvexBinaryCrossentropy(reg_lambda, C, radius_constant)

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      # [] for compatibility with tensorflow loss calculation
      {'testcase_name': 'both positive',
       'logits': [10000],
       'y_true': [1],
       'result': 0,
      },
      {'testcase_name': 'positive gradient negative logits',
       'logits': [-10000],
       'y_true': [1],
       'result': 10000,
      },
      {'testcase_name': 'positivee gradient positive logits',
       'logits': [10000],
       'y_true': [0],
       'result': 10000,
      },
      {'testcase_name': 'both negative',
       'logits': [-10000],
       'y_true': [0],
       'result': 0
      },
  ])
  def test_calculation(self, logits, y_true, result):
    """Test the call method to ensure it returns the correct value.

    Args:
      logits: unscaled output of model
      y_true: label
      result: correct loss calculation value
    """
    logits = tf.Variable(logits, False, dtype=tf.float32)
    y_true = tf.Variable(y_true, False, dtype=tf.float32)
    loss = StrongConvexBinaryCrossentropy(0.00001, 1, 1)
    loss = loss(y_true, logits)
    self.assertEqual(loss.numpy(), result)

  @parameterized.named_parameters([
      {'testcase_name': 'beta',
       'init_args': [1, 1, 1],
       'fn': 'beta',
       'args': [1],
       'result': tf.constant(2, dtype=tf.float32)
      },
      {'testcase_name': 'gamma',
       'fn': 'gamma',
       'init_args': [1, 1, 1],
       'args': [],
       'result': tf.constant(1, dtype=tf.float32),
      },
      {'testcase_name': 'lipchitz constant',
       'fn': 'lipchitz_constant',
       'init_args': [1, 1, 1],
       'args': [1],
       'result': tf.constant(2, dtype=tf.float32),
      },
      {'testcase_name': 'kernel regularizer',
       'fn': 'kernel_regularizer',
       'init_args': [1, 1, 1],
       'args': [],
       'result': L1L2(l2=0.5),
      },
  ])
  def test_fns(self, init_args, fn, args, result):
    """Test that fn of BinaryCrossentropy loss returns the correct result.

    Args:
      init_args: init values for loss instance
      fn: the fn to test
      args: the arguments to above function
      result: the correct result from the fn
    """
    loss = StrongConvexBinaryCrossentropy(*init_args)
    expected = getattr(loss, fn, lambda: 'fn not found')(*args)
    if hasattr(expected, 'numpy') and hasattr(result, 'numpy'):  # both tensor
      expected = expected.numpy()
      result = result.numpy()
    if hasattr(expected, 'l2') and hasattr(result, 'l2'):  # both l2 regularizer
      expected = expected.l2
      result = result.l2
    self.assertEqual(expected, result)

  @parameterized.named_parameters([
      {'testcase_name': 'label_smoothing',
       'init_args': [1, 1, 1, True, 0.1],
       'fn': None,
       'args': None,
       'print_res': 'The impact of label smoothing on privacy is unknown.'
      },
  ])
  def test_prints(self, init_args, fn, args, print_res):
    """Test logger warning from StrongConvexBinaryCrossentropy.

    Args:
      init_args: arguments to init the object with.
      fn: function to test
      args: arguments to above function
      print_res: print result that should have been printed.
    """
    with captured_output() as (out, err):  # pylint: disable=unused-variable
      loss = StrongConvexBinaryCrossentropy(*init_args)
      if fn is not None:
        getattr(loss, fn, lambda *arguments: print('error'))(*args)
    self.assertRegexMatch(err.getvalue().strip(), [print_res])


class HuberTests(keras_parameterized.TestCase):
  """tests for BinaryCrossesntropy StrongConvex loss."""

  @parameterized.named_parameters([
      {'testcase_name': 'normal',
       'reg_lambda': 1,
       'c': 1,
       'radius_constant': 1,
       'delta': 1,
      },
  ])
  def test_init_params(self, reg_lambda, c, radius_constant, delta):
    """Test initialization for given arguments.

    Args:
      reg_lambda: initialization value for reg_lambda arg
      c: initialization value for C arg
      radius_constant: initialization value for radius_constant arg
      delta: the delta parameter for the huber loss
    """
    # test valid domains for each variable
    loss = StrongConvexHuber(reg_lambda, c, radius_constant, delta)
    self.assertIsInstance(loss, StrongConvexHuber)

  @parameterized.named_parameters([
      {'testcase_name': 'negative c',
       'reg_lambda': 1,
       'c': -1,
       'radius_constant': 1,
       'delta': 1
      },
      {'testcase_name': 'negative radius',
       'reg_lambda': 1,
       'c': 1,
       'radius_constant': -1,
       'delta': 1
      },
      {'testcase_name': 'negative lambda',
       'reg_lambda': -1,
       'c': 1,
       'radius_constant': 1,
       'delta': 1
      },
      {'testcase_name': 'negative delta',
       'reg_lambda': 1,
       'c': 1,
       'radius_constant': 1,
       'delta': -1
      },
  ])
  def test_bad_init_params(self, reg_lambda, c, radius_constant, delta):
    """Test invalid domain for given params. Should return ValueError.

    Args:
      reg_lambda: initialization value for reg_lambda arg
      c: initialization value for C arg
      radius_constant: initialization value for radius_constant arg
      delta: the delta parameter for the huber loss
    """
    # test valid domains for each variable
    with self.assertRaises(ValueError):
      StrongConvexHuber(reg_lambda, c, radius_constant, delta)

  # test the bounds and test varied delta's
  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      {'testcase_name': 'delta=1,y_true=1 z>1+h decision boundary',
       'logits': 2.1,
       'y_true': 1,
       'delta': 1,
       'result': 0,
      },
      {'testcase_name': 'delta=1,y_true=1 z<1+h decision boundary',
       'logits': 1.9,
       'y_true': 1,
       'delta': 1,
       'result': 0.01*0.25,
      },
      {'testcase_name': 'delta=1,y_true=1 1-z< h decision boundary',
       'logits': 0.1,
       'y_true': 1,
       'delta': 1,
       'result': 1.9**2 * 0.25,
      },
      {'testcase_name': 'delta=1,y_true=1 z < 1-h decision boundary',
       'logits': -0.1,
       'y_true': 1,
       'delta': 1,
       'result': 1.1,
      },
      {'testcase_name': 'delta=2,y_true=1 z>1+h decision boundary',
       'logits': 3.1,
       'y_true': 1,
       'delta': 2,
       'result': 0,
      },
      {'testcase_name': 'delta=2,y_true=1 z<1+h decision boundary',
       'logits': 2.9,
       'y_true': 1,
       'delta': 2,
       'result': 0.01*0.125,
      },
      {'testcase_name': 'delta=2,y_true=1 1-z < h decision boundary',
       'logits': 1.1,
       'y_true': 1,
       'delta': 2,
       'result': 1.9**2 * 0.125,
      },
      {'testcase_name': 'delta=2,y_true=1 z < 1-h decision boundary',
       'logits': -1.1,
       'y_true': 1,
       'delta': 2,
       'result': 2.1,
      },
      {'testcase_name': 'delta=1,y_true=-1 z>1+h decision boundary',
       'logits': -2.1,
       'y_true': -1,
       'delta': 1,
       'result': 0,
      },
  ])
  def test_calculation(self, logits, y_true, delta, result):
    """Test the call method to ensure it returns the correct value.

    Args:
      logits: unscaled output of model
      y_true: label
      delta: delta value for StrongConvexHuber loss.
      result: correct loss calculation value
    """
    logits = tf.Variable(logits, False, dtype=tf.float32)
    y_true = tf.Variable(y_true, False, dtype=tf.float32)
    loss = StrongConvexHuber(0.00001, 1, 1, delta)
    loss = loss(y_true, logits)
    self.assertAllClose(loss.numpy(), result)

  @parameterized.named_parameters([
      {'testcase_name': 'beta',
       'init_args': [1, 1, 1, 1],
       'fn': 'beta',
       'args': [1],
       'result': tf.Variable(1.5, dtype=tf.float32)
      },
      {'testcase_name': 'gamma',
       'fn': 'gamma',
       'init_args': [1, 1, 1, 1],
       'args': [],
       'result': tf.Variable(1, dtype=tf.float32),
      },
      {'testcase_name': 'lipchitz constant',
       'fn': 'lipchitz_constant',
       'init_args': [1, 1, 1, 1],
       'args': [1],
       'result': tf.Variable(2, dtype=tf.float32),
      },
      {'testcase_name': 'kernel regularizer',
       'fn': 'kernel_regularizer',
       'init_args': [1, 1, 1, 1],
       'args': [],
       'result': L1L2(l2=0.5),
      },
  ])
  def test_fns(self, init_args, fn, args, result):
    """Test that fn of BinaryCrossentropy loss returns the correct result.

    Args:
      init_args: init values for loss instance
      fn: the fn to test
      args: the arguments to above function
      result: the correct result from the fn
    """
    loss = StrongConvexHuber(*init_args)
    expected = getattr(loss, fn, lambda: 'fn not found')(*args)
    if hasattr(expected, 'numpy') and hasattr(result, 'numpy'):  # both tensor
      expected = expected.numpy()
      result = result.numpy()
    if hasattr(expected, 'l2') and hasattr(result, 'l2'):  # both l2 regularizer
      expected = expected.l2
      result = result.l2
    self.assertEqual(expected, result)


if __name__ == '__main__':
  tf.test.main()
