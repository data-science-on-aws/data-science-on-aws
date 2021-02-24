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
"""BoltOn model for Bolt-on method of differentially private ML."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.framework import ops as _ops
from tensorflow.compat.v1.python.keras import optimizers
from tensorflow.compat.v1.python.keras.models import Model
from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexMixin
from tensorflow_privacy.privacy.bolt_on.optimizers import BoltOn


class BoltOnModel(Model):  # pylint: disable=abstract-method
  """BoltOn episilon-delta differential privacy model.

  The privacy guarantees are dependent on the noise that is sampled. Please
  see the paper linked below for more details.

  Uses 4 key steps to achieve privacy guarantees:
  1. Adds noise to weights after training (output perturbation).
  2. Projects weights to R after each batch
  3. Limits learning rate
  4. Use a strongly convex loss function (see compile)

  For more details on the strong convexity requirements, see:
  Bolt-on Differential Privacy for Scalable Stochastic Gradient
  Descent-based Analytics by Xi Wu et al.
  """

  def __init__(self,
               n_outputs,
               seed=1,
               dtype=tf.float32):
    """Private constructor.

    Args:
        n_outputs: number of output classes to predict.
        seed: random seed to use
        dtype: data type to use for tensors
    """
    super(BoltOnModel, self).__init__(name='bolton', dynamic=False)
    if n_outputs <= 0:
      raise ValueError('n_outputs = {0} is not valid. Must be > 0.'.format(
          n_outputs
      ))
    self.n_outputs = n_outputs
    self.seed = seed
    self._layers_instantiated = False
    self._dtype = dtype

  def call(self, inputs):  # pylint: disable=arguments-differ
    """Forward pass of network.

    Args:
        inputs: inputs to neural network

    Returns:
      Output logits for the given inputs.

    """
    return self.output_layer(inputs)

  def compile(self,
              optimizer,
              loss,
              kernel_initializer=tf.initializers.GlorotUniform,
              **kwargs):  # pylint: disable=arguments-differ
    """See super class. Default optimizer used in BoltOn method is SGD.

    Args:
      optimizer: The optimizer to use. This will be automatically wrapped
        with the BoltOn Optimizer.
      loss: The loss function to use. Must be a StrongConvex loss (extend the
        StrongConvexMixin).
      kernel_initializer: The kernel initializer to use for the single layer.
      **kwargs: kwargs to keras Model.compile. See super.
    """
    if not isinstance(loss, StrongConvexMixin):
      raise ValueError('loss function must be a Strongly Convex and therefore '
                       'extend the StrongConvexMixin.')
    if not self._layers_instantiated:  # compile may be called multiple times
      # for instance, if the input/outputs are not defined until fit.
      self.output_layer = tf.keras.layers.Dense(
          self.n_outputs,
          kernel_regularizer=loss.kernel_regularizer(),
          kernel_initializer=kernel_initializer(),
      )
      self._layers_instantiated = True
    if not isinstance(optimizer, BoltOn):
      optimizer = optimizers.get(optimizer)
      optimizer = BoltOn(optimizer, loss)

    super(BoltOnModel, self).compile(optimizer, loss=loss, **kwargs)

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          class_weight=None,
          n_samples=None,
          epsilon=2,
          noise_distribution='laplace',
          steps_per_epoch=None,
          **kwargs):  # pylint: disable=arguments-differ
    """Reroutes to super fit with  BoltOn delta-epsilon privacy requirements.

    Note, inputs must be normalized s.t. ||x|| < 1.
    Requirements are as follows:
      1. Adds noise to weights after training (output perturbation).
      2. Projects weights to R after each batch
      3. Limits learning rate
      4. Use a strongly convex loss function (see compile)
    See super implementation for more details.

    Args:
      x: Inputs to fit on, see super.
      y: Labels to fit on, see super.
      batch_size: The batch size to use for training, see super.
      class_weight: the class weights to be used. Can be a scalar or 1D tensor
                    whose dim == n_classes.
      n_samples: the number of individual samples in x.
      epsilon: privacy parameter, which trades off between utility an privacy.
                See the bolt-on paper for more description.
      noise_distribution: the distribution to pull noise from.
      steps_per_epoch:
      **kwargs: kwargs to keras Model.fit. See super.

    Returns:
      Output from super fit method.
    """
    if class_weight is None:
      class_weight_ = self.calculate_class_weights(class_weight)
    else:
      class_weight_ = class_weight
    if n_samples is not None:
      data_size = n_samples
    elif hasattr(x, 'shape'):
      data_size = x.shape[0]
    elif hasattr(x, '__len__'):
      data_size = len(x)
    else:
      data_size = None
    batch_size_ = self._validate_or_infer_batch_size(batch_size,
                                                     steps_per_epoch,
                                                     x)
    if batch_size_ is None:
      batch_size_ = 32
    # inferring batch_size to be passed to optimizer. batch_size must remain its
    # initial value when passed to super().fit()
    if batch_size_ is None:
      raise ValueError('batch_size: {0} is an '
                       'invalid value'.format(batch_size_))
    if data_size is None:
      raise ValueError('Could not infer the number of samples. Please pass '
                       'this in using n_samples.')
    with self.optimizer(noise_distribution,
                        epsilon,
                        self.layers,
                        class_weight_,
                        data_size,
                        batch_size_) as _:
      out = super(BoltOnModel, self).fit(x=x,
                                         y=y,
                                         batch_size=batch_size,
                                         class_weight=class_weight,
                                         steps_per_epoch=steps_per_epoch,
                                         **kwargs)
    return out

  def fit_generator(self,
                    generator,
                    class_weight=None,
                    noise_distribution='laplace',
                    epsilon=2,
                    n_samples=None,
                    steps_per_epoch=None,
                    **kwargs):  # pylint: disable=arguments-differ
    """Fit with a generator.

    This method is the same as fit except for when the passed dataset
    is a generator. See super method and fit for more details.

    Args:
      generator: Inputs generator following Tensorflow guidelines, see super.
      class_weight: the class weights to be used. Can be a scalar or 1D tensor
                    whose dim == n_classes.
      noise_distribution: the distribution to get noise from.
      epsilon: privacy parameter, which trades off utility and privacy. See
                BoltOn paper for more description.
      n_samples: number of individual samples in x
      steps_per_epoch: Number of steps per training epoch, see super.
      **kwargs: **kwargs

    Returns:
      Output from super fit_generator method.
    """
    if class_weight is None:
      class_weight = self.calculate_class_weights(class_weight)
    if n_samples is not None:
      data_size = n_samples
    elif hasattr(generator, 'shape'):
      data_size = generator.shape[0]
    elif hasattr(generator, '__len__'):
      data_size = len(generator)
    else:
      raise ValueError('The number of samples could not be determined. '
                       'Please make sure that if you are using a generator'
                       'to call this method directly with n_samples kwarg '
                       'passed.')
    batch_size = self._validate_or_infer_batch_size(None, steps_per_epoch,
                                                    generator)
    if batch_size is None:
      batch_size = 32
    with self.optimizer(noise_distribution,
                        epsilon,
                        self.layers,
                        class_weight,
                        data_size,
                        batch_size) as _:
      out = super(BoltOnModel, self).fit_generator(
          generator,
          class_weight=class_weight,
          steps_per_epoch=steps_per_epoch,
          **kwargs)
    return out

  def calculate_class_weights(self,
                              class_weights=None,
                              class_counts=None,
                              num_classes=None):
    """Calculates class weighting to be used in training.

    Args:
      class_weights: str specifying type, array giving weights, or None.
      class_counts: If class_weights is not None, then an array of
                    the number of samples for each class
      num_classes: If class_weights is not None, then the number of
                      classes.
    Returns:
      class_weights as 1D tensor, to be passed to model's fit method.
    """
    # Value checking
    class_keys = ['balanced']
    is_string = False
    if isinstance(class_weights, str):
      is_string = True
      if class_weights not in class_keys:
        raise ValueError('Detected string class_weights with '
                         'value: {0}, which is not one of {1}.'
                         'Please select a valid class_weight type'
                         'or pass an array'.format(class_weights,
                                                   class_keys))
      if class_counts is None:
        raise ValueError('Class counts must be provided if using '
                         'class_weights=%s' % class_weights)
      class_counts_shape = tf.Variable(class_counts,
                                       trainable=False,
                                       dtype=self._dtype).shape
      if len(class_counts_shape) != 1:
        raise ValueError('class counts must be a 1D array.'
                         'Detected: {0}'.format(class_counts_shape))
      if num_classes is None:
        raise ValueError('num_classes must be provided if using '
                         'class_weights=%s' % class_weights)
    elif class_weights is not None:
      if num_classes is None:
        raise ValueError('You must pass a value for num_classes if '
                         'creating an array of class_weights')
    # performing class weight calculation
    if class_weights is None:
      class_weights = 1
    elif is_string and class_weights == 'balanced':
      num_samples = sum(class_counts)
      weighted_counts = tf.dtypes.cast(tf.math.multiply(num_classes,
                                                        class_counts),
                                       self._dtype)
      class_weights = tf.Variable(num_samples, dtype=self._dtype) / \
                      tf.Variable(weighted_counts, dtype=self._dtype)
    else:
      class_weights = _ops.convert_to_tensor_v2(class_weights)
      if len(class_weights.shape) != 1:
        raise ValueError('Detected class_weights shape: {0} instead of '
                         '1D array'.format(class_weights.shape))
      if class_weights.shape[0] != num_classes:
        raise ValueError(
            'Detected array length: {0} instead of: {1}'.format(
                class_weights.shape[0],
                num_classes))
    return class_weights
