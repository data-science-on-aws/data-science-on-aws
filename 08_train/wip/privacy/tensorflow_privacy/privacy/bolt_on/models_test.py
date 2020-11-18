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
"""Unit testing for models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.framework import ops as _ops
from tensorflow.compat.v1.python.keras import keras_parameterized
from tensorflow.compat.v1.python.keras import losses
from tensorflow.compat.v1.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.compat.v1.python.keras.regularizers import L1L2
from tensorflow_privacy.privacy.bolt_on import models
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexMixin
from tensorflow_privacy.privacy.bolt_on.optimizers import BoltOn


class TestLoss(losses.Loss, StrongConvexMixin):
  """Test loss function for testing BoltOn model."""

  def __init__(self, reg_lambda, c_arg, radius_constant, name='test'):
    super(TestLoss, self).__init__(name=name)
    self.reg_lambda = reg_lambda
    self.C = c_arg  # pylint: disable=invalid-name
    self.radius_constant = radius_constant

  def radius(self):
    """Radius, R, of the hypothesis space W.

    W is a convex set that forms the hypothesis space.

    Returns:
      radius
    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def gamma(self):
    """Returns strongly convex parameter, gamma."""
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def beta(self, class_weight):  # pylint: disable=unused-argument
    """Smoothness, beta.

    Args:
      class_weight: the class weights as scalar or 1d tensor, where its
        dimensionality is equal to the number of outputs.

    Returns:
      Beta
    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def lipchitz_constant(self, class_weight):  # pylint: disable=unused-argument
    """Lipchitz constant, L.

    Args:
      class_weight: class weights used

    Returns:
      L
    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def call(self, y_true, y_pred):
    """Loss function that is minimized at the mean of the input points."""
    return 0.5 * tf.reduce_sum(
        tf.math.squared_difference(y_true, y_pred),
        axis=1
    )

  def max_class_weight(self, class_weight):
    """the maximum weighting in class weights (max value) as a scalar tensor.

    Args:
      class_weight: class weights used

    Returns:
      maximum class weighting as tensor scalar
    """
    if class_weight is None:
      return 1
    raise ValueError('')

  def kernel_regularizer(self):
    """Returns the kernel_regularizer to be used.

    Any subclass should override this method if they want a kernel_regularizer
    (if required for the loss function to be StronglyConvex.
    """
    return L1L2(l2=self.reg_lambda)


class TestOptimizer(OptimizerV2):
  """Test optimizer used for testing BoltOn model."""

  def __init__(self):
    super(TestOptimizer, self).__init__('test')

  def compute_gradients(self):
    return 0

  def get_config(self):
    return {}

  def _create_slots(self, var):
    pass

  def _resource_apply_dense(self, grad, handle):
    return grad

  def _resource_apply_sparse(self, grad, handle, indices):
    return grad


class InitTests(keras_parameterized.TestCase):
  """Tests for keras model initialization."""

  @parameterized.named_parameters([
      {'testcase_name': 'normal',
       'n_outputs': 1,
      },
      {'testcase_name': 'many outputs',
       'n_outputs': 100,
      },
  ])
  def test_init_params(self, n_outputs):
    """Test initialization of BoltOnModel.

    Args:
        n_outputs: number of output neurons
    """
    # test valid domains for each variable
    clf = models.BoltOnModel(n_outputs)
    self.assertIsInstance(clf, models.BoltOnModel)

  @parameterized.named_parameters([
      {'testcase_name': 'invalid n_outputs',
       'n_outputs': -1,
      },
  ])
  def test_bad_init_params(self, n_outputs):
    """test bad initializations of BoltOnModel that should raise errors.

    Args:
        n_outputs: number of output neurons
    """
    # test invalid domains for each variable, especially noise
    with self.assertRaises(ValueError):
      models.BoltOnModel(n_outputs)

  @parameterized.named_parameters([
      {'testcase_name': 'string compile',
       'n_outputs': 1,
       'loss': TestLoss(1, 1, 1),
       'optimizer': 'adam',
      },
      {'testcase_name': 'test compile',
       'n_outputs': 100,
       'loss': TestLoss(1, 1, 1),
       'optimizer': TestOptimizer(),
      },
  ])
  def test_compile(self, n_outputs, loss, optimizer):
    """Test compilation of BoltOnModel.

    Args:
      n_outputs: number of output neurons
      loss: instantiated TestLoss instance
      optimizer: instantiated TestOptimizer instance
    """
    # test compilation of valid tf.optimizer and tf.loss
    with self.cached_session():
      clf = models.BoltOnModel(n_outputs)
      clf.compile(optimizer, loss)
      self.assertEqual(clf.loss, loss)

  @parameterized.named_parameters([
      {'testcase_name': 'Not strong loss',
       'n_outputs': 1,
       'loss': losses.BinaryCrossentropy(),
       'optimizer': 'adam',
      },
      {'testcase_name': 'Not valid optimizer',
       'n_outputs': 1,
       'loss': TestLoss(1, 1, 1),
       'optimizer': 'ada',
      }
  ])
  def test_bad_compile(self, n_outputs, loss, optimizer):
    """test bad compilations of BoltOnModel that should raise errors.

    Args:
      n_outputs: number of output neurons
      loss: instantiated TestLoss instance
      optimizer: instantiated TestOptimizer instance
    """
    # test compilaton of invalid tf.optimizer and non instantiated loss.
    with self.cached_session():
      with self.assertRaises((ValueError, AttributeError)):
        clf = models.BoltOnModel(n_outputs)
        clf.compile(optimizer, loss)


def _cat_dataset(n_samples, input_dim, n_classes, batch_size, generator=False):
  """Creates a categorically encoded dataset.

  Creates a categorically encoded dataset (y is categorical).
  returns the specified dataset either as a static array or as a generator.
  Will have evenly split samples across each output class.
  Each output class will be a different point in the input space.

  Args:
    n_samples: number of rows
    input_dim: input dimensionality
    n_classes: output dimensionality
    batch_size: The desired batch_size
    generator: False for array, True for generator

  Returns:
    X as (n_samples, input_dim), Y as (n_samples, n_outputs)
  """
  x_stack = []
  y_stack = []
  for i_class in range(n_classes):
    x_stack.append(
        tf.constant(1*i_class, tf.float32, (n_samples, input_dim))
    )
    y_stack.append(
        tf.constant(i_class, tf.float32, (n_samples, n_classes))
    )
  x_set, y_set = tf.stack(x_stack), tf.stack(y_stack)
  if generator:
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_set, y_set)
    )
    dataset = dataset.batch(batch_size=batch_size)
    return dataset
  return x_set, y_set


def _do_fit(n_samples,
            input_dim,
            n_outputs,
            epsilon,
            generator,
            batch_size,
            reset_n_samples,
            optimizer,
            loss,
            distribution='laplace'):
  """Instantiate necessary components for fitting and perform a model fit.

  Args:
    n_samples: number of samples in dataset
    input_dim: the sample dimensionality
    n_outputs: number of output neurons
    epsilon: privacy parameter
    generator: True to create a generator, False to use an iterator
    batch_size: batch_size to use
    reset_n_samples: True to set _samples to None prior to fitting.
                      False does nothing
    optimizer: instance of TestOptimizer
    loss: instance of TestLoss
    distribution: distribution to get noise from.

  Returns:
    BoltOnModel instsance
  """
  clf = models.BoltOnModel(n_outputs)
  clf.compile(optimizer, loss)
  if generator:
    x = _cat_dataset(
        n_samples,
        input_dim,
        n_outputs,
        batch_size,
        generator=generator
    )
    y = None
    # x = x.batch(batch_size)
    x = x.shuffle(n_samples//2)
    batch_size = None
    if reset_n_samples:
      n_samples = None
    clf.fit_generator(x,
                      n_samples=n_samples,
                      noise_distribution=distribution,
                      epsilon=epsilon)
  else:
    x, y = _cat_dataset(
        n_samples,
        input_dim,
        n_outputs,
        batch_size,
        generator=generator)
    if reset_n_samples:
      n_samples = None
    clf.fit(x,
            y,
            batch_size=batch_size,
            n_samples=n_samples,
            noise_distribution=distribution,
            epsilon=epsilon)
  return clf


class FitTests(keras_parameterized.TestCase):
  """Test cases for keras model fitting."""

  # @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      {'testcase_name': 'iterator fit',
       'generator': False,
       'reset_n_samples': True,
      },
      {'testcase_name': 'iterator fit no samples',
       'generator': False,
       'reset_n_samples': True,
      },
      {'testcase_name': 'generator fit',
       'generator': True,
       'reset_n_samples': False,
      },
      {'testcase_name': 'with callbacks',
       'generator': True,
       'reset_n_samples': False,
      },
  ])
  def test_fit(self, generator, reset_n_samples):
    """Tests fitting of BoltOnModel.

    Args:
      generator: True for generator test, False for iterator test.
      reset_n_samples: True to reset the n_samples to None, False does nothing
    """
    loss = TestLoss(1, 1, 1)
    optimizer = BoltOn(TestOptimizer(), loss)
    n_classes = 2
    input_dim = 5
    epsilon = 1
    batch_size = 1
    n_samples = 10
    clf = _do_fit(
        n_samples,
        input_dim,
        n_classes,
        epsilon,
        generator,
        batch_size,
        reset_n_samples,
        optimizer,
        loss,
    )
    self.assertEqual(hasattr(clf, 'layers'), True)

  @parameterized.named_parameters([
      {'testcase_name': 'generator fit',
       'generator': True,
      },
  ])
  def test_fit_gen(self, generator):
    """Tests the fit_generator method of BoltOnModel.

    Args:
      generator: True to test with a generator dataset
    """
    loss = TestLoss(1, 1, 1)
    optimizer = TestOptimizer()
    n_classes = 2
    input_dim = 5
    batch_size = 1
    n_samples = 10
    clf = models.BoltOnModel(n_classes)
    clf.compile(optimizer, loss)
    x = _cat_dataset(
        n_samples,
        input_dim,
        n_classes,
        batch_size,
        generator=generator
    )
    x = x.batch(batch_size)
    x = x.shuffle(n_samples // 2)
    clf.fit_generator(x, n_samples=n_samples)
    self.assertEqual(hasattr(clf, 'layers'), True)

  @parameterized.named_parameters([
      {'testcase_name': 'iterator no n_samples',
       'generator': True,
       'reset_n_samples': True,
       'distribution': 'laplace'
      },
      {'testcase_name': 'invalid distribution',
       'generator': True,
       'reset_n_samples': True,
       'distribution': 'not_valid'
      },
  ])
  def test_bad_fit(self, generator, reset_n_samples, distribution):
    """Tests fitting with invalid parameters, which should raise an error.

    Args:
      generator: True to test with generator, False is iterator
      reset_n_samples: True to reset the n_samples param to None prior to
        passing it to fit
      distribution: distribution to get noise from.
    """
    with self.assertRaises(ValueError):
      loss = TestLoss(1, 1, 1)
      optimizer = TestOptimizer()
      n_classes = 2
      input_dim = 5
      epsilon = 1
      batch_size = 1
      n_samples = 10
      _do_fit(
          n_samples,
          input_dim,
          n_classes,
          epsilon,
          generator,
          batch_size,
          reset_n_samples,
          optimizer,
          loss,
          distribution
      )

  @parameterized.named_parameters([
      {'testcase_name': 'None class_weights',
       'class_weights': None,
       'class_counts': None,
       'num_classes': None,
       'result': 1},
      {'testcase_name': 'class weights array',
       'class_weights': [1, 1],
       'class_counts': [1, 1],
       'num_classes': 2,
       'result': [1, 1]},
      {'testcase_name': 'class weights balanced',
       'class_weights': 'balanced',
       'class_counts': [1, 1],
       'num_classes': 2,
       'result': [1, 1]},
  ])
  def test_class_calculate(self,
                           class_weights,
                           class_counts,
                           num_classes,
                           result):
    """Tests the BOltonModel calculate_class_weights method.

    Args:
      class_weights: the class_weights to use
      class_counts: count of number of samples for each class
      num_classes: number of outputs neurons
      result: expected result
    """
    clf = models.BoltOnModel(1, 1)
    expected = clf.calculate_class_weights(class_weights,
                                           class_counts,
                                           num_classes)

    if hasattr(expected, 'numpy'):
      expected = expected.numpy()
    self.assertAllEqual(
        expected,
        result
    )
  @parameterized.named_parameters([
      {'testcase_name': 'class weight not valid str',
       'class_weights': 'not_valid',
       'class_counts': 1,
       'num_classes': 1,
       'err_msg': 'Detected string class_weights with value: not_valid'},
      {'testcase_name': 'no class counts',
       'class_weights': 'balanced',
       'class_counts': None,
       'num_classes': 1,
       'err_msg': 'Class counts must be provided if '
                  'using class_weights=balanced'},
      {'testcase_name': 'no num classes',
       'class_weights': 'balanced',
       'class_counts': [1],
       'num_classes': None,
       'err_msg': 'num_classes must be provided if '
                  'using class_weights=balanced'},
      {'testcase_name': 'class counts not array',
       'class_weights': 'balanced',
       'class_counts': 1,
       'num_classes': None,
       'err_msg': 'class counts must be a 1D array.'},
      {'testcase_name': 'class counts array, no num classes',
       'class_weights': [1],
       'class_counts': None,
       'num_classes': None,
       'err_msg': 'You must pass a value for num_classes if '
                  'creating an array of class_weights'},
      {'testcase_name': 'class counts array, improper shape',
       'class_weights': [[1], [1]],
       'class_counts': None,
       'num_classes': 2,
       'err_msg': 'Detected class_weights shape'},
      {'testcase_name': 'class counts array, wrong number classes',
       'class_weights': [1, 1, 1],
       'class_counts': None,
       'num_classes': 2,
       'err_msg': 'Detected array length:'},
  ])

  def test_class_errors(self,
                        class_weights,
                        class_counts,
                        num_classes,
                        err_msg):
    """Tests the BOltonModel calculate_class_weights method.

    This test passes invalid params which should raise the expected errors.

    Args:
      class_weights: the class_weights to use.
      class_counts: count of number of samples for each class.
      num_classes: number of outputs neurons.
      err_msg: The expected error message.
    """
    clf = models.BoltOnModel(1, 1)
    with self.assertRaisesRegexp(ValueError, err_msg):  # pylint: disable=deprecated-method
      clf.calculate_class_weights(class_weights,
                                  class_counts,
                                  num_classes)


if __name__ == '__main__':
  tf.test.main()
