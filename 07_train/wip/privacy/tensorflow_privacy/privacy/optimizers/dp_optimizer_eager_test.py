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
"""Tests for differentially private optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.optimizers import dp_optimizer


class DPOptimizerEagerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    tf.enable_eager_execution()
    super(DPOptimizerEagerTest, self).setUp()

  def _loss_fn(self, val0, val1):
    return 0.5 * tf.reduce_sum(
        input_tensor=tf.math.squared_difference(val0, val1), axis=1)

  @parameterized.named_parameters(
      ('DPGradientDescent 1', dp_optimizer.DPGradientDescentOptimizer, 1,
       [-2.5, -2.5]),
      ('DPGradientDescent 2', dp_optimizer.DPGradientDescentOptimizer, 2,
       [-2.5, -2.5]),
      ('DPGradientDescent 4', dp_optimizer.DPGradientDescentOptimizer, 4,
       [-2.5, -2.5]),
      ('DPAdagrad 1', dp_optimizer.DPAdagradOptimizer, 1, [-2.5, -2.5]),
      ('DPAdagrad 2', dp_optimizer.DPAdagradOptimizer, 2, [-2.5, -2.5]),
      ('DPAdagrad 4', dp_optimizer.DPAdagradOptimizer, 4, [-2.5, -2.5]),
      ('DPAdam 1', dp_optimizer.DPAdamOptimizer, 1, [-2.5, -2.5]),
      ('DPAdam 2', dp_optimizer.DPAdamOptimizer, 2, [-2.5, -2.5]),
      ('DPAdam 4', dp_optimizer.DPAdamOptimizer, 4, [-2.5, -2.5]))
  def testBaseline(self, cls, num_microbatches, expected_answer):
    with tf.GradientTape(persistent=True) as gradient_tape:
      var0 = tf.Variable([1.0, 2.0])
      data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])

      dp_sum_query = gaussian_query.GaussianSumQuery(1.0e9, 0.0)
      dp_sum_query = privacy_ledger.QueryWithLedger(
          dp_sum_query, 1e6, num_microbatches / 1e6)

      opt = cls(
          dp_sum_query,
          num_microbatches=num_microbatches,
          learning_rate=2.0)

      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))

      # Expected gradient is sum of differences divided by number of
      # microbatches.
      grads_and_vars = opt.compute_gradients(
          lambda: self._loss_fn(var0, data0), [var0],
          gradient_tape=gradient_tape)
      self.assertAllCloseAccordingToType(expected_answer, grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer.DPGradientDescentOptimizer),
      ('DPAdagrad', dp_optimizer.DPAdagradOptimizer),
      ('DPAdam', dp_optimizer.DPAdamOptimizer))
  def testClippingNorm(self, cls):
    with tf.GradientTape(persistent=True) as gradient_tape:
      var0 = tf.Variable([0.0, 0.0])
      data0 = tf.Variable([[3.0, 4.0], [6.0, 8.0]])

      dp_sum_query = gaussian_query.GaussianSumQuery(1.0, 0.0)
      dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, 1e6, 1 / 1e6)

      opt = cls(dp_sum_query, num_microbatches=1, learning_rate=2.0)

      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([0.0, 0.0], self.evaluate(var0))

      # Expected gradient is sum of differences.
      grads_and_vars = opt.compute_gradients(
          lambda: self._loss_fn(var0, data0), [var0],
          gradient_tape=gradient_tape)
      self.assertAllCloseAccordingToType([-0.6, -0.8], grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent', dp_optimizer.DPGradientDescentOptimizer),
      ('DPAdagrad', dp_optimizer.DPAdagradOptimizer),
      ('DPAdam', dp_optimizer.DPAdamOptimizer))
  def testNoiseMultiplier(self, cls):
    with tf.GradientTape(persistent=True) as gradient_tape:
      var0 = tf.Variable([0.0])
      data0 = tf.Variable([[0.0]])

      dp_sum_query = gaussian_query.GaussianSumQuery(4.0, 8.0)
      dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, 1e6, 1 / 1e6)

      opt = cls(dp_sum_query, num_microbatches=1, learning_rate=2.0)

      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([0.0], self.evaluate(var0))

      grads = []
      for _ in range(1000):
        grads_and_vars = opt.compute_gradients(
            lambda: self._loss_fn(var0, data0), [var0],
            gradient_tape=gradient_tape)
        grads.append(grads_and_vars[0][0])

      # Test standard deviation is close to l2_norm_clip * noise_multiplier.
      self.assertNear(np.std(grads), 2.0 * 4.0, 0.5)


if __name__ == '__main__':
  tf.test.main()
