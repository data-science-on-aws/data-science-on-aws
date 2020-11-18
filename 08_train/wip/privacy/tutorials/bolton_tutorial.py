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
"""Tutorial for bolt_on module, the model and the optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf  # pylint: disable=wrong-import-position
from tensorflow_privacy.privacy.bolt_on import losses  # pylint: disable=wrong-import-position
from tensorflow_privacy.privacy.bolt_on import models  # pylint: disable=wrong-import-position
from tensorflow_privacy.privacy.bolt_on.optimizers import BoltOn  # pylint: disable=wrong-import-position
# -------
# First, we will create a binary classification dataset with a single output
# dimension. The samples for each label are repeated data points at different
# points in space.
# -------
# Parameters for dataset
n_samples = 10
input_dim = 2
n_outputs = 1
# Create binary classification dataset:
x_stack = [tf.constant(-1, tf.float32, (n_samples, input_dim)),
           tf.constant(1, tf.float32, (n_samples, input_dim))]
y_stack = [tf.constant(0, tf.float32, (n_samples, 1)),
           tf.constant(1, tf.float32, (n_samples, 1))]
x, y = tf.concat(x_stack, 0), tf.concat(y_stack, 0)
print(x.shape, y.shape)
generator = tf.data.Dataset.from_tensor_slices((x, y))
generator = generator.batch(10)
generator = generator.shuffle(10)
# -------
# First, we will explore using the pre - built BoltOnModel, which is a thin
# wrapper around a Keras Model using a single - layer neural network.
# It automatically uses the BoltOn Optimizer which encompasses all the logic
# required for the BoltOn Differential Privacy method.
# -------
bolt = models.BoltOnModel(n_outputs)  # tell the model how many outputs we have.
# -------
# Now, we will pick our optimizer and Strongly Convex Loss function. The loss
# must extend from StrongConvexMixin and implement the associated methods.Some
# existing loss functions are pre - implemented in bolt_on.loss
# -------
optimizer = tf.optimizers.SGD()
reg_lambda = 1
C = 1
radius_constant = 1
loss = losses.StrongConvexBinaryCrossentropy(reg_lambda, C, radius_constant)
# -------
# For simplicity, we pick all parameters of the StrongConvexBinaryCrossentropy
# to be 1; these are all tunable and their impact can be read in losses.
# StrongConvexBinaryCrossentropy.We then compile the model with the chosen
# optimizer and loss, which will automatically wrap the chosen optimizer with
# the BoltOn Optimizer, ensuring the required components function as required
# for privacy guarantees.
# -------
bolt.compile(optimizer, loss)
# -------
# To fit the model, the optimizer will require additional information about
# the dataset and model.These parameters are:
# 1. the class_weights used
# 2. the number of samples in the dataset
# 3. the batch size which the model will try to infer, if possible.  If not,
# you will be required to pass these explicitly to the fit method.
#
# As well, there are two privacy parameters than can be altered:
# 1. epsilon, a float
# 2. noise_distribution, a valid string indicating the distriution to use (must
# be implemented)
#
# The BoltOnModel offers a helper method,.calculate_class_weight to aid in
# class_weight calculation.
# required parameters
# -------
class_weight = None  # default, use .calculate_class_weight for other values
batch_size = None  # default, if it cannot be inferred, specify this
n_samples = None  # default, if it cannot be iferred, specify this
# privacy parameters
epsilon = 2
noise_distribution = 'laplace'

bolt.fit(x,
         y,
         epsilon=epsilon,
         class_weight=class_weight,
         batch_size=batch_size,
         n_samples=n_samples,
         noise_distribution=noise_distribution,
         epochs=2)
# -------
# We may also train a generator object, or try different optimizers and loss
# functions. Below, we will see that we must pass the number of samples as the
# fit method is unable to infer it for a generator.
# -------
optimizer2 = tf.optimizers.Adam()
bolt.compile(optimizer2, loss)
# required parameters
class_weight = None  # default, use .calculate_class_weight for other values
batch_size = None  # default, if it cannot be inferred, specify this
n_samples = None  # default, if it cannot be iferred, specify this
# privacy parameters
epsilon = 2
noise_distribution = 'laplace'
try:
  bolt.fit(generator,
           epsilon=epsilon,
           class_weight=class_weight,
           batch_size=batch_size,
           n_samples=n_samples,
           noise_distribution=noise_distribution,
           verbose=0)
except ValueError as e:
  print(e)
# -------
# And now, re running with the parameter set.
# -------
n_samples = 20
bolt.fit_generator(generator,
                   epsilon=epsilon,
                   class_weight=class_weight,
                   n_samples=n_samples,
                   noise_distribution=noise_distribution,
                   verbose=0)
# -------
# You don't have to use the BoltOn model to use the BoltOn method.
# There are only a few requirements:
# 1. make sure any requirements from the loss are implemented in the model.
# 2. instantiate the optimizer and use it as a context around the fit operation.
# -------
# -------------------- Part 2, using the Optimizer

# -------
# Here, we create our own model and setup the BoltOn optimizer.
# -------


class TestModel(tf.keras.Model):  # pylint: disable=abstract-method

  def __init__(self, reg_layer, number_of_outputs=1):
    super(TestModel, self).__init__(name='test')
    self.output_layer = tf.keras.layers.Dense(number_of_outputs,
                                              kernel_regularizer=reg_layer)

  def call(self, inputs):  # pylint: disable=arguments-differ
    return self.output_layer(inputs)


optimizer = tf.optimizers.SGD()
loss = losses.StrongConvexBinaryCrossentropy(reg_lambda, C, radius_constant)
optimizer = BoltOn(optimizer, loss)
# -------
# Now, we instantiate our model and check for 1. Since our loss requires L2
# regularization over the kernel, we will pass it to the model.
# -------
n_outputs = 1  # parameter for model and optimizer context.
test_model = TestModel(loss.kernel_regularizer(), n_outputs)
test_model.compile(optimizer, loss)
# -------
# We comply with 2., and use the BoltOn Optimizer as a context around the fit
# method.
# -------
# parameters for context
noise_distribution = 'laplace'
epsilon = 2
class_weights = 1  # Previously, the fit method auto-detected the class_weights.
# Here, we need to pass the class_weights explicitly. 1 is the same as None.
n_samples = 20
batch_size = 5

with optimizer(
    noise_distribution=noise_distribution,
    epsilon=epsilon,
    layers=test_model.layers,
    class_weights=class_weights,
    n_samples=n_samples,
    batch_size=batch_size
) as _:
  test_model.fit(x, y, batch_size=batch_size, epochs=2)
