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
"""BoltOn Optimizer for Bolt-on method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.compat.v1.python.ops import math_ops
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexMixin

_accepted_distributions = ['laplace']  # implemented distributions for noising


class GammaBetaDecreasingStep(
    optimizer_v2.learning_rate_schedule.LearningRateSchedule):
  """Computes LR as minimum of 1/beta and 1/(gamma * step) at each step.

  This is a required step for privacy guarantees.
  """

  def __init__(self):
    self.is_init = False
    self.beta = None
    self.gamma = None

  def __call__(self, step):
    """Computes and returns the learning rate.

    Args:
      step: the current iteration number

    Returns:
      decayed learning rate to minimum of 1/beta and 1/(gamma * step) as per
      the BoltOn privacy requirements.
    """
    if not self.is_init:
      raise AttributeError('Please initialize the {0} Learning Rate Scheduler.'
                           'This is performed automatically by using the '
                           '{1} as a context manager, '
                           'as desired'.format(self.__class__.__name__,
                                               BoltOn.__class__.__name__
                                              )
                          )
    dtype = self.beta.dtype
    one = tf.constant(1, dtype)
    return tf.math.minimum(tf.math.reduce_min(one/self.beta),
                           one/(self.gamma*math_ops.cast(step, dtype))
                          )

  def get_config(self):
    """Return config to setup the learning rate scheduler."""
    return {'beta': self.beta, 'gamma': self.gamma}

  def initialize(self, beta, gamma):
    """Setups scheduler with beta and gamma values from the loss function.

    Meant to be used with .fit as the loss params may depend on values passed to
    fit.

    Args:
      beta: Smoothness value. See StrongConvexMixin
      gamma: Strong Convexity parameter. See StrongConvexMixin.
    """
    self.is_init = True
    self.beta = beta
    self.gamma = gamma

  def de_initialize(self):
    """De initialize post fit, as another fit call may use other parameters."""
    self.is_init = False
    self.beta = None
    self.gamma = None


class BoltOn(optimizer_v2.OptimizerV2):
  """Wrap another tf optimizer with BoltOn privacy protocol.

  BoltOn optimizer wraps another tf optimizer to be used
  as the visible optimizer to the tf model. No matter the optimizer
  passed, "BoltOn" enables the bolt-on model to control the learning rate
  based on the strongly convex loss.

  To use the BoltOn method, you must:
  1. instantiate it with an instantiated tf optimizer and StrongConvexLoss.
  2. use it as a context manager around your .fit method internals.

  This can be accomplished by the following:
  optimizer = tf.optimizers.SGD()
  loss = privacy.bolt_on.losses.StrongConvexBinaryCrossentropy()
  bolton = BoltOn(optimizer, loss)
  with bolton(*args) as _:
    model.fit()
  The args required for the context manager can be found in the __call__
  method.

  For more details on the strong convexity requirements, see:
  Bolt-on Differential Privacy for Scalable Stochastic Gradient
  Descent-based Analytics by Xi Wu et. al.
  """

  def __init__(self,  # pylint: disable=super-init-not-called
               optimizer,
               loss,
               dtype=tf.float32,
              ):
    """Constructor.

    Args:
      optimizer: Optimizer_v2 or subclass to be used as the optimizer
        (wrapped).
      loss: StrongConvexLoss function that the model is being compiled with.
      dtype: dtype
    """

    if not isinstance(loss, StrongConvexMixin):
      raise ValueError('loss function must be a Strongly Convex and therefore '
                       'extend the StrongConvexMixin.')
    self._private_attributes = [
        '_internal_optimizer',
        'dtype',
        'noise_distribution',
        'epsilon',
        'loss',
        'class_weights',
        'input_dim',
        'n_samples',
        'layers',
        'batch_size',
        '_is_init',
    ]
    self._internal_optimizer = optimizer
    self.learning_rate = GammaBetaDecreasingStep()  # use the BoltOn Learning
    # rate scheduler, as required for privacy guarantees. This will still need
    # to get values from the loss function near the time that .fit is called
    # on the model (when this optimizer will be called as a context manager)
    self.dtype = dtype
    self.loss = loss
    self._is_init = False

  def get_config(self):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    return self._internal_optimizer.get_config()

  def project_weights_to_r(self, force=False):
    """Normalize the weights to the R-ball.

    Args:
      force: True to normalize regardless of previous weight values.
        False to check if weights > R-ball and only normalize then.

    Raises:
      Exception: If not called from inside this optimizer context.
    """
    if not self._is_init:
      raise Exception('This method must be called from within the optimizer\'s '
                      'context.')
    radius = self.loss.radius()
    for layer in self.layers:
      weight_norm = tf.norm(layer.kernel, axis=0)
      if force:
        layer.kernel = layer.kernel / (weight_norm / radius)
      else:
        layer.kernel = tf.cond(
            tf.reduce_sum(tf.cast(weight_norm > radius, dtype=self.dtype)) > 0,
            lambda k=layer.kernel, w=weight_norm, r=radius: k / (w / r),  # pylint: disable=cell-var-from-loop
            lambda k=layer.kernel: k  # pylint: disable=cell-var-from-loop
        )

  def get_noise(self, input_dim, output_dim):
    """Sample noise to be added to weights for privacy guarantee.

    Args:
      input_dim: the input dimensionality for the weights
      output_dim: the output dimensionality for the weights

    Returns:
      Noise in shape of layer's weights to be added to the weights.

    Raises:
      Exception: If not called from inside this optimizer's context.
    """
    if not self._is_init:
      raise Exception('This method must be called from within the optimizer\'s '
                      'context.')
    loss = self.loss
    distribution = self.noise_distribution.lower()
    if distribution == _accepted_distributions[0]:  # laplace
      per_class_epsilon = self.epsilon / (output_dim)
      l2_sensitivity = (2 *
                        loss.lipchitz_constant(self.class_weights)) / \
                       (loss.gamma() * self.n_samples * self.batch_size)
      unit_vector = tf.random.normal(shape=(input_dim, output_dim),
                                     mean=0,
                                     seed=1,
                                     stddev=1.0,
                                     dtype=self.dtype)
      unit_vector = unit_vector / tf.math.sqrt(
          tf.reduce_sum(tf.math.square(unit_vector), axis=0)
      )

      beta = l2_sensitivity / per_class_epsilon
      alpha = input_dim  # input_dim
      gamma = tf.random.gamma([output_dim],
                              alpha,
                              beta=1 / beta,
                              seed=1,
                              dtype=self.dtype
                             )
      return unit_vector * gamma
    raise NotImplementedError('Noise distribution: {0} is not '
                              'a valid distribution'.format(distribution))

  def from_config(self, *args, **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    return self._internal_optimizer.from_config(*args, **kwargs)

  def __getattr__(self, name):
    """Get attr.

    return _internal_optimizer off self instance, and everything else
    from the _internal_optimizer instance.

    Args:
      name: Name of attribute to get from this or aggregate optimizer.

    Returns:
      attribute from BoltOn if specified to come from self, else
      from _internal_optimizer.
    """
    if name == '_private_attributes' or name in self._private_attributes:
      return getattr(self, name)
    optim = object.__getattribute__(self, '_internal_optimizer')
    try:
      return object.__getattribute__(optim, name)
    except AttributeError:
      raise AttributeError(
          "Neither '{0}' nor '{1}' object has attribute '{2}'"
          "".format(self.__class__.__name__,
                    self._internal_optimizer.__class__.__name__,
                    name)
          )

  def __setattr__(self, key, value):
    """Set attribute to self instance if its the internal optimizer.

    Reroute everything else to the _internal_optimizer.

    Args:
      key: attribute name
      value: attribute value
    """
    if key == '_private_attributes':
      object.__setattr__(self, key, value)
    elif key in self._private_attributes:
      object.__setattr__(self, key, value)
    else:
      setattr(self._internal_optimizer, key, value)

  def _resource_apply_dense(self, *args, **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    return self._internal_optimizer._resource_apply_dense(*args, **kwargs)  # pylint: disable=protected-access

  def _resource_apply_sparse(self, *args, **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    return self._internal_optimizer._resource_apply_sparse(*args, **kwargs)  # pylint: disable=protected-access

  def get_updates(self, loss, params):
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    out = self._internal_optimizer.get_updates(loss, params)
    self.project_weights_to_r()
    return out

  def apply_gradients(self, *args, **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    out = self._internal_optimizer.apply_gradients(*args, **kwargs)
    self.project_weights_to_r()
    return out

  def minimize(self, *args, **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    out = self._internal_optimizer.minimize(*args, **kwargs)
    self.project_weights_to_r()
    return out

  def _compute_gradients(self, *args, **kwargs):  # pylint: disable=arguments-differ,protected-access
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    return self._internal_optimizer._compute_gradients(*args, **kwargs)  # pylint: disable=protected-access

  def get_gradients(self, *args, **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to _internal_optimizer. See super/_internal_optimizer."""
    return self._internal_optimizer.get_gradients(*args, **kwargs)

  def __enter__(self):
    """Context manager call at the beginning of with statement.

    Returns:
      self, to be used in context manager
    """
    self._is_init = True
    return self

  def __call__(self,
               noise_distribution,
               epsilon,
               layers,
               class_weights,
               n_samples,
               batch_size):
    """Accepts required values for bolton method from context entry point.

    Stores them on the optimizer for use throughout fitting.

    Args:
      noise_distribution: the noise distribution to pick.
        see _accepted_distributions and get_noise for possible values.
      epsilon: privacy parameter. Lower gives more privacy but less utility.
      layers: list of Keras/Tensorflow layers. Can be found as model.layers
      class_weights: class_weights used, which may either be a scalar or 1D
        tensor with dim == n_classes.
      n_samples: number of rows/individual samples in the training set
      batch_size: batch size used.

    Returns:
      self, to be used by the __enter__ method for context.
    """
    if epsilon <= 0:
      raise ValueError('Detected epsilon: {0}. '
                       'Valid range is 0 < epsilon <inf'.format(epsilon))
    if noise_distribution not in _accepted_distributions:
      raise ValueError('Detected noise distribution: {0} not one of: {1} valid'
                       'distributions'.format(noise_distribution,
                                              _accepted_distributions))
    self.noise_distribution = noise_distribution
    self.learning_rate.initialize(self.loss.beta(class_weights),
                                  self.loss.gamma())
    self.epsilon = tf.constant(epsilon, dtype=self.dtype)
    self.class_weights = tf.constant(class_weights, dtype=self.dtype)
    self.n_samples = tf.constant(n_samples, dtype=self.dtype)
    self.layers = layers
    self.batch_size = tf.constant(batch_size, dtype=self.dtype)
    return self

  def __exit__(self, *args):
    """Exit call from with statement.

    Used to:
    1.reset the model and fit parameters passed to the optimizer
      to enable the BoltOn Privacy guarantees. These are reset to ensure
      that any future calls to fit with the same instance of the optimizer
      will properly error out.

    2.call post-fit methods normalizing/projecting the model weights and
      adding noise to the weights.

    Args:
      *args: encompasses the type, value, and traceback values which are unused.
    """
    self.project_weights_to_r(True)
    for layer in self.layers:
      input_dim = layer.kernel.shape[0]
      output_dim = layer.units
      noise = self.get_noise(input_dim,
                             output_dim,
                            )
      layer.kernel = tf.math.add(layer.kernel, noise)
    self.noise_distribution = None
    self.learning_rate.de_initialize()
    self.epsilon = -1
    self.batch_size = -1
    self.class_weights = None
    self.n_samples = None
    self.input_dim = None
    self.layers = None
    self._is_init = False
