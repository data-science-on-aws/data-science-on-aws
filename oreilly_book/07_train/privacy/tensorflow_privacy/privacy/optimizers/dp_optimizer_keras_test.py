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
"""Tests for dp_optimizer_keras.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras


class DPOptimizerComputeGradientsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for _compute_gradients method."""

  def _loss(self, val0, val1):
    """Loss function whose derivative w.r.t val1 is val1 - val0."""
    return 0.5 * tf.reduce_sum(
        input_tensor=tf.math.squared_difference(val0, val1), axis=1)

  # Parameters for testing: optimizer, num_microbatches, expected gradient for
  # var0, expected gradient for var1.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer_keras.DPKerasAdamOptimizer, 2,
       [-2.5, -2.5], [-0.5]),
      ('DPAdagrad 4', dp_optimizer_keras.DPKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
  )
  def testBaselineWithCallableLoss(self, cls, num_microbatches, expected_grad0,
                                   expected_grad1):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])

    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = opt._compute_gradients(loss, [var0, var1])
    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  # Parameters for testing: optimizer, num_microbatches, expected gradient for
  # var0, expected gradient for var1.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1,
       [-2.5, -2.5], [-0.5]),
      ('DPAdam 2', dp_optimizer_keras.DPKerasAdamOptimizer, 2, [-2.5, -2.5
                                                               ], [-0.5]),
      ('DPAdagrad 4', dp_optimizer_keras.DPKerasAdagradOptimizer, 4,
       [-2.5, -2.5], [-0.5]),
  )
  def testBaselineWithTensorLoss(self, cls, num_microbatches, expected_grad0,
                                 expected_grad1):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])

    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    tape = tf.GradientTape()
    with tape:
      loss = self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = opt._compute_gradients(
        loss, [var0, var1], tape=tape)
    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer_keras.DPKerasSGDOptimizer),)
  def testClippingNorm(self, cls):
    var0 = tf.Variable([0.0, 0.0])
    data0 = tf.Variable([[3.0, 4.0], [6.0, 8.0]])

    opt = cls(
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0)
    # Expected gradient is sum of differences.
    grads_and_vars = opt._compute_gradients(loss, [var0])
    self.assertAllCloseAccordingToType([-0.6, -0.8], grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent 2 4 1', dp_optimizer_keras.DPKerasSGDOptimizer, 2.0,
       4.0, 1),
      ('DPGradientDescent 4 1 4', dp_optimizer_keras.DPKerasSGDOptimizer, 4.0,
       1.0, 4),
  )
  def testNoiseMultiplier(self, cls, l2_norm_clip, noise_multiplier,
                          num_microbatches):
    var0 = tf.Variable(tf.zeros([1000], dtype=tf.float32))
    data0 = tf.Variable(tf.zeros([16, 1000], dtype=tf.float32))

    opt = cls(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0)
    grads_and_vars = opt._compute_gradients(loss, [var0])
    grads = grads_and_vars[0][0].numpy()

    # Test standard deviation is close to l2_norm_clip * noise_multiplier.
    self.assertNear(
        np.std(grads), l2_norm_clip * noise_multiplier / num_microbatches, 0.5)

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer_keras.DPKerasSGDOptimizer),
      ('DPAdagrad', dp_optimizer_keras.DPKerasAdagradOptimizer),
      ('DPAdam', dp_optimizer_keras.DPKerasAdamOptimizer))
  def testAssertOnNoCallOfComputeGradients(self, cls):
    """Tests that assertion fails when DP gradients are not computed."""
    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    with self.assertRaises(AssertionError):
      grads_and_vars = tf.Variable([0.0])
      opt.apply_gradients(grads_and_vars)

    # Expect no exception if _compute_gradients is called.
    var0 = tf.Variable([0.0])
    data0 = tf.Variable([[0.0]])
    loss = lambda: self._loss(data0, var0)
    grads_and_vars = opt._compute_gradients(loss, [var0])
    opt.apply_gradients(grads_and_vars)


class DPOptimizerGetGradientsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for get_gradient method.

  Since get_gradients must run in graph mode, the method is tested within
  the Estimator framework.
  """

  def _make_linear_model_fn(self, opt_cls, l2_norm_clip, noise_multiplier,
                            num_microbatches, learning_rate):
    """Returns a model function for a linear regressor."""

    def linear_model_fn(features, labels, mode):
      layer = tf.keras.layers.Dense(
          1,
          activation='linear',
          name='dense',
          kernel_initializer='zeros',
          bias_initializer='zeros')
      preds = layer.apply(features)

      vector_loss = 0.5 * tf.math.squared_difference(labels, preds)
      scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

      optimizer = opt_cls(
          l2_norm_clip=l2_norm_clip,
          noise_multiplier=noise_multiplier,
          num_microbatches=num_microbatches,
          learning_rate=learning_rate)

      params = layer.trainable_weights
      global_step = tf.compat.v1.train.get_global_step()
      train_op = tf.group(
          optimizer.get_updates(loss=vector_loss, params=params),
          [tf.compat.v1.assign_add(global_step, 1)])
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=scalar_loss, train_op=train_op)

    return linear_model_fn

  # Parameters for testing: optimizer, num_microbatches.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),
      ('DPGradientDescent 2', dp_optimizer_keras.DPKerasSGDOptimizer, 2),
      ('DPGradientDescent 4', dp_optimizer_keras.DPKerasSGDOptimizer, 4),)
  def testBaseline(self, cls, num_microbatches):
    """Tests that DP optimizers work with tf.estimator."""

    linear_regressor = tf.estimator.Estimator(
        model_fn=self._make_linear_model_fn(cls, 100.0, 0.0, num_microbatches,
                                            0.05))

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)

    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    def train_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    linear_regressor.train(input_fn=train_input_fn, steps=125)

    self.assertAllClose(
        linear_regressor.get_variable_value('dense/kernel'),
        true_weights,
        atol=0.05)
    self.assertAllClose(
        linear_regressor.get_variable_value('dense/bias'), true_bias, atol=0.05)

  # Parameters for testing: optimizer, num_microbatches.
  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),)
  def testClippingNorm(self, cls, num_microbatches):
    """Tests that DP optimizers work with tf.estimator."""

    true_weights = np.array([[6.0], [0.0], [0], [0]]).astype(np.float32)
    true_bias = np.array([0]).astype(np.float32)

    train_data = np.array([[1.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    train_labels = np.matmul(train_data, true_weights) + true_bias

    def train_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(1)

    unclipped_linear_regressor = tf.estimator.Estimator(
        model_fn=self._make_linear_model_fn(cls, 1.0e9, 0.0, num_microbatches,
                                            1.0))
    unclipped_linear_regressor.train(input_fn=train_input_fn, steps=1)

    kernel_value = unclipped_linear_regressor.get_variable_value('dense/kernel')
    bias_value = unclipped_linear_regressor.get_variable_value('dense/bias')
    global_norm = np.linalg.norm(np.concatenate((kernel_value, [bias_value])))

    clipped_linear_regressor = tf.estimator.Estimator(
        model_fn=self._make_linear_model_fn(cls, 1.0, 0.0, num_microbatches,
                                            1.0))
    clipped_linear_regressor.train(input_fn=train_input_fn, steps=1)

    self.assertAllClose(
        clipped_linear_regressor.get_variable_value('dense/kernel'),
        kernel_value / global_norm,
        atol=0.001)
    self.assertAllClose(
        clipped_linear_regressor.get_variable_value('dense/bias'),
        bias_value / global_norm,
        atol=0.001)

  # Parameters for testing: optimizer, l2_norm_clip, noise_multiplier,
  # num_microbatches.
  @parameterized.named_parameters(
      ('DPGradientDescent 2 4 1', dp_optimizer_keras.DPKerasSGDOptimizer, 2.0,
       4.0, 1),
      ('DPGradientDescent 3 2 4', dp_optimizer_keras.DPKerasSGDOptimizer, 3.0,
       2.0, 4),
      ('DPGradientDescent 8 6 8', dp_optimizer_keras.DPKerasSGDOptimizer, 8.0,
       6.0, 8),
  )
  def testNoiseMultiplier(self, cls, l2_norm_clip, noise_multiplier,
                          num_microbatches):
    """Tests that DP optimizers work with tf.estimator."""

    linear_regressor = tf.estimator.Estimator(
        model_fn=self._make_linear_model_fn(
            cls,
            l2_norm_clip,
            noise_multiplier,
            num_microbatches,
            learning_rate=1.0))

    true_weights = np.zeros((1000, 1), dtype=np.float32)
    true_bias = np.array([0.0]).astype(np.float32)

    train_data = np.zeros((16, 1000), dtype=np.float32)
    train_labels = np.matmul(train_data, true_weights) + true_bias

    def train_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(16)

    linear_regressor.train(input_fn=train_input_fn, steps=1)

    kernel_value = linear_regressor.get_variable_value('dense/kernel')
    self.assertNear(
        np.std(kernel_value),
        l2_norm_clip * noise_multiplier / num_microbatches, 0.5)

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer_keras.DPKerasSGDOptimizer),
      ('DPAdagrad', dp_optimizer_keras.DPKerasAdagradOptimizer),
      ('DPAdam', dp_optimizer_keras.DPKerasAdamOptimizer))
  def testAssertOnNoCallOfGetGradients(self, cls):
    """Tests that assertion fails when DP gradients are not computed."""
    opt = cls(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0)

    with self.assertRaises(AssertionError):
      grads_and_vars = tf.Variable([0.0])
      opt.apply_gradients(grads_and_vars)


if __name__ == '__main__':
  tf.test.main()
