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
"""Loss functions for BoltOn method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.framework import ops as _ops
from tensorflow.compat.v1.python.keras import losses
from tensorflow.compat.v1.python.keras.regularizers import L1L2
from tensorflow.compat.v1.python.keras.utils import losses_utils
from tensorflow.compat.v1.python.platform import tf_logging as logging


class StrongConvexMixin:  # pylint: disable=old-style-class
  """Strong Convex Mixin base class.

  Strong Convex Mixin base class for any loss function that will be used with
  BoltOn model. Subclasses must be strongly convex and implement the
  associated constants. They must also conform to the requirements of tf losses
  (see super class).

  For more details on the strong convexity requirements, see:
  Bolt-on Differential Privacy for Scalable Stochastic Gradient
  Descent-based Analytics by Xi Wu et. al.
  """

  def radius(self):
    """Radius, R, of the hypothesis space W.

    W is a convex set that forms the hypothesis space.

    Returns:
      R
    """
    raise NotImplementedError("Radius not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def gamma(self):
    """Returns strongly convex parameter, gamma."""
    raise NotImplementedError("Gamma not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def beta(self, class_weight):
    """Smoothness, beta.

    Args:
      class_weight: the class weights as scalar or 1d tensor, where its
        dimensionality is equal to the number of outputs.

    Returns:
      Beta
    """
    raise NotImplementedError("Beta not implemented for StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def lipchitz_constant(self, class_weight):
    """Lipchitz constant, L.

    Args:
      class_weight: class weights used

    Returns: L
    """
    raise NotImplementedError("lipchitz constant not implemented for "
                              "StrongConvex Loss"
                              "function: %s" % str(self.__class__.__name__))

  def kernel_regularizer(self):
    """Returns the kernel_regularizer to be used.

    Any subclass should override this method if they want a kernel_regularizer
    (if required for the loss function to be StronglyConvex.
    """
    return None

  def max_class_weight(self, class_weight, dtype):
    """The maximum weighting in class weights (max value) as a scalar tensor.

    Args:
      class_weight: class weights used
      dtype: the data type for tensor conversions.

    Returns:
      maximum class weighting as tensor scalar
    """
    class_weight = _ops.convert_to_tensor_v2(class_weight, dtype)
    return tf.math.reduce_max(class_weight)


class StrongConvexHuber(losses.Loss, StrongConvexMixin):
  """Strong Convex version of Huber loss using l2 weight regularization."""

  def __init__(self,
               reg_lambda,
               c_arg,
               radius_constant,
               delta,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               dtype=tf.float32):
    """Constructor.

    Args:
      reg_lambda: Weight regularization constant
      c_arg: Penalty parameter C of the loss term
      radius_constant: constant defining the length of the radius
      delta: delta value in huber loss.  When to switch from quadratic to
        absolute deviation.
      reduction: reduction type to use. See super class
      dtype: tf datatype to use for tensor conversions.

    Returns:
      Loss values per sample.
    """
    if c_arg <= 0:
      raise ValueError("c: {0}, should be >= 0".format(c_arg))
    if reg_lambda <= 0:
      raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
    if radius_constant <= 0:
      raise ValueError("radius_constant: {0}, should be >= 0".format(
          radius_constant
      ))
    if delta <= 0:
      raise ValueError("delta: {0}, should be >= 0".format(
          delta
      ))
    self.C = c_arg  # pylint: disable=invalid-name
    self.delta = delta
    self.radius_constant = radius_constant
    self.dtype = dtype
    self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
    super(StrongConvexHuber, self).__init__(
        name="strongconvexhuber",
        reduction=reduction,
    )

  def call(self, y_true, y_pred):
    """Computes loss.

    Args:
      y_true: Ground truth values. One hot encoded using -1 and 1.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    h = self.delta
    z = y_pred * y_true
    one = tf.constant(1, dtype=self.dtype)
    four = tf.constant(4, dtype=self.dtype)

    if z > one + h:  # pylint: disable=no-else-return
      return _ops.convert_to_tensor_v2(0, dtype=self.dtype)
    elif tf.math.abs(one - z) <= h:
      return one / (four * h) * tf.math.pow(one + h - z, 2)
    return one - z

  def radius(self):
    """See super class."""
    return self.radius_constant / self.reg_lambda

  def gamma(self):
    """See super class."""
    return self.reg_lambda

  def beta(self, class_weight):
    """See super class."""
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    delta = _ops.convert_to_tensor_v2(self.delta,
                                      dtype=self.dtype
                                     )
    return self.C * max_class_weight / (delta *
                                        tf.constant(2, dtype=self.dtype)) + \
           self.reg_lambda

  def lipchitz_constant(self, class_weight):
    """See super class."""
    # if class_weight is provided,
    # it should be a vector of the same size of number of classes
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    lc = self.C * max_class_weight + \
         self.reg_lambda * self.radius()
    return lc

  def kernel_regularizer(self):
    """Return l2 loss using 0.5*reg_lambda as the l2 term (as desired).

    L2 regularization is required for this loss function to be strongly convex.

    Returns:
      The L2 regularizer layer for this loss function, with regularizer constant
      set to half the 0.5 * reg_lambda.
    """
    return L1L2(l2=self.reg_lambda/2)


class StrongConvexBinaryCrossentropy(
    losses.BinaryCrossentropy,
    StrongConvexMixin
):
  """Strongly Convex BinaryCrossentropy loss using l2 weight regularization."""

  def __init__(self,
               reg_lambda,
               c_arg,
               radius_constant,
               from_logits=True,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               dtype=tf.float32):
    """StrongConvexBinaryCrossentropy class.

    Args:
      reg_lambda: Weight regularization constant
      c_arg: Penalty parameter C of the loss term
      radius_constant: constant defining the length of the radius
      from_logits: True if the input are unscaled logits. False if they are
        already scaled.
      label_smoothing: amount of smoothing to perform on labels
        relaxation of trust in labels, e.g. (1 -> 1-x, 0 -> 0+x). Note, the
        impact of this parameter's effect on privacy is not known and thus the
        default should be used.
      reduction: reduction type to use. See super class
      dtype: tf datatype to use for tensor conversions.
    """
    if label_smoothing != 0:
      logging.warning("The impact of label smoothing on privacy is unknown. "
                      "Use label smoothing at your own risk as it may not "
                      "guarantee privacy.")

    if reg_lambda <= 0:
      raise ValueError("reg lambda: {0} must be positive".format(reg_lambda))
    if c_arg <= 0:
      raise ValueError("c: {0}, should be >= 0".format(c_arg))
    if radius_constant <= 0:
      raise ValueError("radius_constant: {0}, should be >= 0".format(
          radius_constant
      ))
    self.dtype = dtype
    self.C = c_arg  # pylint: disable=invalid-name
    self.reg_lambda = tf.constant(reg_lambda, dtype=self.dtype)
    super(StrongConvexBinaryCrossentropy, self).__init__(
        reduction=reduction,
        name="strongconvexbinarycrossentropy",
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )
    self.radius_constant = radius_constant

  def call(self, y_true, y_pred):
    """Computes loss.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    loss = super(StrongConvexBinaryCrossentropy, self).call(y_true, y_pred)
    loss = loss * self.C
    return loss

  def radius(self):
    """See super class."""
    return self.radius_constant / self.reg_lambda

  def gamma(self):
    """See super class."""
    return self.reg_lambda

  def beta(self, class_weight):
    """See super class."""
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    return self.C * max_class_weight + self.reg_lambda

  def lipchitz_constant(self, class_weight):
    """See super class."""
    max_class_weight = self.max_class_weight(class_weight, self.dtype)
    return self.C * max_class_weight + self.reg_lambda * self.radius()

  def kernel_regularizer(self):
    """Return l2 loss using 0.5*reg_lambda as the l2 term (as desired).

    L2 regularization is required for this loss function to be strongly convex.

    Returns:
      The L2 regularizer layer for this loss function, with regularizer constant
      set to half the 0.5 * reg_lambda.
    """
    return L1L2(l2=self.reg_lambda/2)
