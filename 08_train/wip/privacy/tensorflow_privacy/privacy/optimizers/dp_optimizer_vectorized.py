# Copyright 2020, The TensorFlow Authors.
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
"""Vectorized differentially private optimizers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow.compat.v1 as tf

AdagradOptimizer = tf.train.AdagradOptimizer
AdamOptimizer = tf.train.AdamOptimizer
GradientDescentOptimizer = tf.train.GradientDescentOptimizer
parent_code = tf.train.Optimizer.compute_gradients.__code__
GATE_OP = tf.train.Optimizer.GATE_OP  # pylint: disable=invalid-name


def make_vectorized_optimizer_class(cls):
  """Constructs a vectorized DP optimizer class from an existing one."""
  child_code = cls.compute_gradients.__code__
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients)
        noise_multiplier: Ratio of the standard deviation to the clipping norm
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None):
      if callable(loss):
        # TF is running in Eager mode
        raise NotImplementedError('Vectorized optimizer unavailable for TF2.')
      else:
        # TF is running in graph mode, check we did not receive a gradient tape.
        if gradient_tape:
          raise ValueError('When in graph mode, a tape should not be passed.')

        batch_size = tf.shape(input=loss)[0]
        if self._num_microbatches is None:
          self._num_microbatches = batch_size

        # Note: it would be closer to the correct i.i.d. sampling of records if
        # we sampled each microbatch from the appropriate binomial distribution,
        # although that still wouldn't be quite correct because it would be
        # sampling from the dataset without replacement.
        microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])

        if var_list is None:
          var_list = (
              tf.trainable_variables() + tf.get_collection(
                  tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        def process_microbatch(microbatch_loss):
          """Compute clipped grads for one microbatch."""
          microbatch_loss = tf.reduce_mean(input_tensor=microbatch_loss)
          grads, _ = zip(*super(DPOptimizerClass, self).compute_gradients(
              microbatch_loss,
              var_list,
              gate_gradients,
              aggregation_method,
              colocate_gradients_with_ops,
              grad_loss))
          grads_list = [
              g if g is not None else tf.zeros_like(v)
              for (g, v) in zip(list(grads), var_list)
          ]
          # Clip gradients to have L2 norm of l2_norm_clip.
          # Here, we use TF primitives rather than the built-in
          # tf.clip_by_global_norm() so that operations can be vectorized
          # across microbatches.
          grads_flat = tf.nest.flatten(grads_list)
          squared_l2_norms = [
              tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
          ]
          global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
          div = tf.maximum(global_norm / self._l2_norm_clip, 1.)
          clipped_flat = [g / div for g in grads_flat]
          clipped_grads = tf.nest.pack_sequence_as(grads_list, clipped_flat)
          return clipped_grads

        clipped_grads = tf.vectorized_map(process_microbatch, microbatch_losses)

        def reduce_noise_normalize_batch(stacked_grads):
          summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
          noise_stddev = self._l2_norm_clip * self._noise_multiplier
          noise = tf.random.normal(
              tf.shape(input=summed_grads), stddev=noise_stddev)
          noised_grads = summed_grads + noise
          return noised_grads / tf.cast(self._num_microbatches, tf.float32)

        final_grads = tf.nest.map_structure(reduce_noise_normalize_batch,
                                            clipped_grads)

        return list(zip(final_grads, var_list))

  return DPOptimizerClass


VectorizedDPAdagrad = make_vectorized_optimizer_class(AdagradOptimizer)
VectorizedDPAdam = make_vectorized_optimizer_class(AdamOptimizer)
VectorizedDPSGD = make_vectorized_optimizer_class(GradientDescentOptimizer)
