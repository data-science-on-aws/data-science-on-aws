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
"""Unit testing for optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python import ops as _ops
from tensorflow.compat.v1.python.framework import test_util
from tensorflow.compat.v1.python.keras import keras_parameterized
from tensorflow.compat.v1.python.keras import losses
from tensorflow.compat.v1.python.keras.initializers import constant
from tensorflow.compat.v1.python.keras.models import Model
from tensorflow.compat.v1.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.compat.v1.python.keras.regularizers import L1L2
from tensorflow.compat.v1.python.platform import test
from tensorflow_privacy.privacy.bolt_on import optimizers as opt
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexMixin


class TestModel(Model):  # pylint: disable=abstract-method
  """BoltOn episilon-delta model.

  Uses 4 key steps to achieve privacy guarantees:
  1. Adds noise to weights after training (output perturbation).
  2. Projects weights to R after each batch
  3. Limits learning rate
  4. Use a strongly convex loss function (see compile)

  For more details on the strong convexity requirements, see:
  Bolt-on Differential Privacy for Scalable Stochastic Gradient
  Descent-based Analytics by Xi Wu et. al.
  """

  def __init__(self, n_outputs=2, input_shape=(16,), init_value=2):
    """Constructor.

    Args:
      n_outputs: number of output neurons
      input_shape:
      init_value:
    """
    super(TestModel, self).__init__(name='bolton', dynamic=False)
    self.n_outputs = n_outputs
    self.layer_input_shape = input_shape
    self.output_layer = tf.keras.layers.Dense(
        self.n_outputs,
        input_shape=self.layer_input_shape,
        kernel_regularizer=L1L2(l2=1),
        kernel_initializer=constant(init_value),
    )


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
      a tensor
    """
    return _ops.convert_to_tensor_v2(self.radius_constant, dtype=tf.float32)

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
      constant L
    """
    return _ops.convert_to_tensor_v2(1, dtype=tf.float32)

  def call(self, y_true, y_pred):
    """Loss function that is minimized at the mean of the input points."""
    return 0.5 * tf.reduce_sum(
        tf.math.squared_difference(y_true, y_pred),
        axis=1
    )

  def max_class_weight(self, class_weight, dtype=tf.float32):
    """the maximum weighting in class weights (max value) as a scalar tensor.

    Args:
      class_weight: class weights used
      dtype: the data type for tensor conversions.

    Returns:
      maximum class weighting as tensor scalar
    """
    if class_weight is None:
      return 1
    raise NotImplementedError('')

  def kernel_regularizer(self):
    """Returns the kernel_regularizer to be used.

    Any subclass should override this method if they want a kernel_regularizer
    (if required for the loss function to be StronglyConvex.
    """
    return L1L2(l2=self.reg_lambda)


class TestOptimizer(OptimizerV2):
  """Optimizer used for testing the BoltOn optimizer."""

  def __init__(self):
    super(TestOptimizer, self).__init__('test')
    self.not_private = 'test'
    self.iterations = tf.constant(1, dtype=tf.float32)
    self._iterations = tf.constant(1, dtype=tf.float32)

  def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
    return 'test'

  def get_config(self):
    return 'test'

  def from_config(self, config, custom_objects=None):
    return 'test'

  def _create_slots(self):
    return 'test'

  def _resource_apply_dense(self, grad, handle):
    return 'test'

  def _resource_apply_sparse(self, grad, handle, indices):
    return 'test'

  def get_updates(self, loss, params):
    return 'test'

  def apply_gradients(self, grads_and_vars, name=None):
    return 'test'

  def minimize(self, loss, var_list, grad_loss=None, name=None):
    return 'test'

  def get_gradients(self, loss, params):
    return 'test'

  def limit_learning_rate(self):
    return 'test'


class BoltonOptimizerTest(keras_parameterized.TestCase):
  """BoltOn Optimizer tests."""
  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      {'testcase_name': 'getattr',
       'fn': '__getattr__',
       'args': ['dtype'],
       'result': tf.float32,
       'test_attr': None},
      {'testcase_name': 'project_weights_to_r',
       'fn': 'project_weights_to_r',
       'args': ['dtype'],
       'result': None,
       'test_attr': ''},
  ])

  def test_fn(self, fn, args, result, test_attr):
    """test that a fn of BoltOn optimizer is working as expected.

    Args:
      fn: method of Optimizer to test
      args: args to optimizer fn
      result: the expected result
      test_attr: None if the fn returns the test result. Otherwise, this is
        the attribute of BoltOn to check against result with.

    """
    tf.random.set_seed(1)
    loss = TestLoss(1, 1, 1)
    bolton = opt.BoltOn(TestOptimizer(), loss)
    model = TestModel(1)
    model.layers[0].kernel = \
      model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                          model.n_outputs))
    bolton._is_init = True  # pylint: disable=protected-access
    bolton.layers = model.layers
    bolton.epsilon = 2
    bolton.noise_distribution = 'laplace'
    bolton.n_outputs = 1
    bolton.n_samples = 1
    res = getattr(bolton, fn, None)(*args)
    if test_attr is not None:
      res = getattr(bolton, test_attr, None)
    if hasattr(res, 'numpy') and hasattr(result, 'numpy'):  # both tensors/not
      res = res.numpy()
      result = result.numpy()
    self.assertEqual(res, result)

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      {'testcase_name': '1 value project to r=1',
       'r': 1,
       'init_value': 2,
       'shape': (1,),
       'n_out': 1,
       'result': [[1]]},
      {'testcase_name': '2 value project to r=1',
       'r': 1,
       'init_value': 2,
       'shape': (2,),
       'n_out': 1,
       'result': [[0.707107], [0.707107]]},
      {'testcase_name': '1 value project to r=2',
       'r': 2,
       'init_value': 3,
       'shape': (1,),
       'n_out': 1,
       'result': [[2]]},
      {'testcase_name': 'no project',
       'r': 2,
       'init_value': 1,
       'shape': (1,),
       'n_out': 1,
       'result': [[1]]},
  ])
  def test_project(self, r, shape, n_out, init_value, result):
    """test that a fn of BoltOn optimizer is working as expected.

    Args:
      r: Radius value for StrongConvex loss function.
      shape: input_dimensionality
      n_out: output dimensionality
      init_value: the initial value for 'constant' kernel initializer
      result: the expected output after projection.
    """
    tf.random.set_seed(1)
    def project_fn(r):
      loss = TestLoss(1, 1, r)
      bolton = opt.BoltOn(TestOptimizer(), loss)
      model = TestModel(n_out, shape, init_value)
      model.compile(bolton, loss)
      model.layers[0].kernel = \
        model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                            model.n_outputs))
      bolton._is_init = True  # pylint: disable=protected-access
      bolton.layers = model.layers
      bolton.epsilon = 2
      bolton.noise_distribution = 'laplace'
      bolton.n_outputs = 1
      bolton.n_samples = 1
      bolton.project_weights_to_r()
      return _ops.convert_to_tensor_v2(bolton.layers[0].kernel, tf.float32)
    res = project_fn(r)
    self.assertAllClose(res, result)

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters([
      {'testcase_name': 'normal values',
       'epsilon': 2,
       'noise': 'laplace',
       'class_weights': 1},
  ])
  def test_context_manager(self, noise, epsilon, class_weights):
    """Tests the context manager functionality of the optimizer.

    Args:
      noise: noise distribution to pick
      epsilon: epsilon privacy parameter to use
      class_weights: class_weights to use
    """
    @tf.function
    def test_run():
      loss = TestLoss(1, 1, 1)
      bolton = opt.BoltOn(TestOptimizer(), loss)
      model = TestModel(1, (1,), 1)
      model.compile(bolton, loss)
      model.layers[0].kernel = \
        model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                            model.n_outputs))
      with bolton(noise, epsilon, model.layers, class_weights, 1, 1) as _:
        pass
      return _ops.convert_to_tensor_v2(bolton.epsilon, dtype=tf.float32)
    epsilon = test_run()
    self.assertEqual(epsilon.numpy(), -1)

  @parameterized.named_parameters([
      {'testcase_name': 'invalid noise',
       'epsilon': 1,
       'noise': 'not_valid',
       'err_msg': 'Detected noise distribution: not_valid not one of:'},
      {'testcase_name': 'invalid epsilon',
       'epsilon': -1,
       'noise': 'laplace',
       'err_msg': 'Detected epsilon: -1. Valid range is 0 < epsilon <inf'},
  ])
  def test_context_domains(self, noise, epsilon, err_msg):
    """Tests the context domains.

    Args:
      noise: noise distribution to pick
      epsilon: epsilon privacy parameter to use
      err_msg: the expected error message

    """

    @tf.function
    def test_run(noise, epsilon):
      loss = TestLoss(1, 1, 1)
      bolton = opt.BoltOn(TestOptimizer(), loss)
      model = TestModel(1, (1,), 1)
      model.compile(bolton, loss)
      model.layers[0].kernel = \
        model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                            model.n_outputs))
      with bolton(noise, epsilon, model.layers, 1, 1, 1) as _:
        pass
    with self.assertRaisesRegexp(ValueError, err_msg):  # pylint: disable=deprecated-method
      test_run(noise, epsilon)

  @parameterized.named_parameters([
      {'testcase_name': 'fn: get_noise',
       'fn': 'get_noise',
       'args': [1, 1],
       'err_msg': 'This method must be called from within the '
                  'optimizer\'s context'},
  ])
  def test_not_in_context(self, fn, args, err_msg):
    """Tests that the expected functions raise errors when not in context.

    Args:
        fn: the function to test
        args: the arguments for said function
        err_msg: expected error message
    """
    def test_run(fn, args):
      loss = TestLoss(1, 1, 1)
      bolton = opt.BoltOn(TestOptimizer(), loss)
      model = TestModel(1, (1,), 1)
      model.compile(bolton, loss)
      model.layers[0].kernel = \
        model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                            model.n_outputs))
      getattr(bolton, fn)(*args)

    with self.assertRaisesRegexp(Exception, err_msg):  # pylint: disable=deprecated-method
      test_run(fn, args)

  @parameterized.named_parameters([
      {'testcase_name': 'fn: get_updates',
       'fn': 'get_updates',
       'args': [0, 0]},
      {'testcase_name': 'fn: get_config',
       'fn': 'get_config',
       'args': []},
      {'testcase_name': 'fn: from_config',
       'fn': 'from_config',
       'args': [0]},
      {'testcase_name': 'fn: _resource_apply_dense',
       'fn': '_resource_apply_dense',
       'args': [1, 1]},
      {'testcase_name': 'fn: _resource_apply_sparse',
       'fn': '_resource_apply_sparse',
       'args': [1, 1, 1]},
      {'testcase_name': 'fn: apply_gradients',
       'fn': 'apply_gradients',
       'args': [1]},
      {'testcase_name': 'fn: minimize',
       'fn': 'minimize',
       'args': [1, 1]},
      {'testcase_name': 'fn: _compute_gradients',
       'fn': '_compute_gradients',
       'args': [1, 1]},
      {'testcase_name': 'fn: get_gradients',
       'fn': 'get_gradients',
       'args': [1, 1]},
  ])
  def test_rerouted_function(self, fn, args):
    """Tests rerouted function.

    Tests that a method of the internal optimizer is correctly routed from
    the BoltOn instance to the internal optimizer instance (TestOptimizer,
    here).

    Args:
      fn: fn to test
      args: arguments to that fn
    """
    loss = TestLoss(1, 1, 1)
    optimizer = TestOptimizer()
    bolton = opt.BoltOn(optimizer, loss)
    model = TestModel(3)
    model.compile(optimizer, loss)
    model.layers[0].kernel = \
        model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                            model.n_outputs))
    model.layers[0].kernel = \
      model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                          model.n_outputs))
    bolton._is_init = True  # pylint: disable=protected-access
    bolton.layers = model.layers
    bolton.epsilon = 2
    bolton.noise_distribution = 'laplace'
    bolton.n_outputs = 1
    bolton.n_samples = 1
    self.assertEqual(
        getattr(bolton, fn, lambda: 'fn not found')(*args),
        'test'
    )

  @parameterized.named_parameters([
      {'testcase_name': 'fn: project_weights_to_r',
       'fn': 'project_weights_to_r',
       'args': []},
      {'testcase_name': 'fn: get_noise',
       'fn': 'get_noise',
       'args': [1, 1]},
  ])
  def test_not_reroute_fn(self, fn, args):
    """Test function is not rerouted.

    Test that a fn that should not be rerouted to the internal optimizer is
    in fact not rerouted.

    Args:
      fn: fn to test
      args: arguments to that fn
    """
    def test_run(fn, args):
      loss = TestLoss(1, 1, 1)
      bolton = opt.BoltOn(TestOptimizer(), loss)
      model = TestModel(1, (1,), 1)
      model.compile(bolton, loss)
      model.layers[0].kernel = \
        model.layers[0].kernel_initializer((model.layer_input_shape[0],
                                            model.n_outputs))
      bolton._is_init = True  # pylint: disable=protected-access
      bolton.noise_distribution = 'laplace'
      bolton.epsilon = 1
      bolton.layers = model.layers
      bolton.class_weights = 1
      bolton.n_samples = 1
      bolton.batch_size = 1
      bolton.n_outputs = 1
      res = getattr(bolton, fn, lambda: 'test')(*args)
      if res != 'test':
        res = 1
      else:
        res = 0
      return _ops.convert_to_tensor_v2(res, dtype=tf.float32)
    self.assertNotEqual(test_run(fn, args), 0)

  @parameterized.named_parameters([
      {'testcase_name': 'attr: _iterations',
       'attr': '_iterations'}
  ])
  def test_reroute_attr(self, attr):
    """Test a function is rerouted.

    Test that attribute of internal optimizer is correctly rerouted to the
    internal optimizer.

    Args:
      attr: attribute to test
    """
    loss = TestLoss(1, 1, 1)
    internal_optimizer = TestOptimizer()
    optimizer = opt.BoltOn(internal_optimizer, loss)
    self.assertEqual(getattr(optimizer, attr),
                     getattr(internal_optimizer, attr))

  @parameterized.named_parameters([
      {'testcase_name': 'attr does not exist',
       'attr': '_not_valid'}
  ])
  def test_attribute_error(self, attr):
    """Test rerouting of attributes.

    Test that attribute of internal optimizer is correctly rerouted to the
    internal optimizer

    Args:
      attr: attribute to test
    """
    loss = TestLoss(1, 1, 1)
    internal_optimizer = TestOptimizer()
    optimizer = opt.BoltOn(internal_optimizer, loss)
    with self.assertRaises(AttributeError):
      getattr(optimizer, attr)


class SchedulerTest(keras_parameterized.TestCase):
  """GammaBeta Scheduler tests."""

  @parameterized.named_parameters([
      {'testcase_name': 'not in context',
       'err_msg': 'Please initialize the GammaBetaDecreasingStep Learning Rate'
                  ' Scheduler'
      }
  ])
  def test_bad_call(self, err_msg):
    """Test attribute of internal opt correctly rerouted to the internal opt.

    Args:
      err_msg: The expected error message from the scheduler bad call.
    """
    scheduler = opt.GammaBetaDecreasingStep()
    with self.assertRaisesRegexp(Exception, err_msg):  # pylint: disable=deprecated-method
      scheduler(1)

  @parameterized.named_parameters([
      {'testcase_name': 'step 1',
       'step': 1,
       'res': 0.5},
      {'testcase_name': 'step 2',
       'step': 2,
       'res': 0.5},
      {'testcase_name': 'step 3',
       'step': 3,
       'res': 0.333333333},
  ])
  def test_call(self, step, res):
    """Test call.

    Test that attribute of internal optimizer is correctly rerouted to the
    internal optimizer

    Args:
      step: step number to 'GammaBetaDecreasingStep' 'Scheduler'.
      res: expected result from call to 'GammaBetaDecreasingStep' 'Scheduler'.
    """
    beta = _ops.convert_to_tensor_v2(2, dtype=tf.float32)
    gamma = _ops.convert_to_tensor_v2(1, dtype=tf.float32)
    scheduler = opt.GammaBetaDecreasingStep()
    scheduler.initialize(beta, gamma)
    step = _ops.convert_to_tensor_v2(step, dtype=tf.float32)
    lr = scheduler(step)
    self.assertAllClose(lr.numpy(), res)


if __name__ == '__main__':
  test.main()
  unittest.main()
