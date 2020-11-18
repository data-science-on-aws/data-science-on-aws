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
"""Common tools for DP-SGD MNIST tutorials."""

# These are not necessary in a Python 3-only module.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def get_cnn_model(features):
  """Given input features, returns the logits from a simple CNN model."""
  input_layer = tf.reshape(features, [-1, 28, 28, 1])
  y = tf.keras.layers.Conv2D(
      16, 8, strides=2, padding='same', activation='relu').apply(input_layer)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Conv2D(
      32, 4, strides=2, padding='valid', activation='relu').apply(y)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Flatten().apply(y)
  y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  logits = tf.keras.layers.Dense(10).apply(y)

  return logits


def make_input_fn(split, input_batch_size=256, repetitions=-1, tpu=False):
  """Make input function on given MNIST split."""

  def input_fn(params=None):
    """A simple input function."""
    batch_size = params.get('batch_size', input_batch_size)

    def parser(example):
      image, label = example['image'], example['label']
      image = tf.cast(image, tf.float32)
      image /= 255.0
      label = tf.cast(label, tf.int32)
      return image, label

    dataset = tfds.load(name='mnist', split=split)
    dataset = dataset.map(parser).shuffle(60000).repeat(repetitions).batch(
        batch_size)
    # If this input function is not meant for TPUs, we can stop here.
    # Otherwise, we need to explicitly set its shape. Note that for unknown
    # reasons, returning the latter format causes performance regression
    # on non-TPUs.
    if not tpu:
      return dataset

    # Give inputs statically known shapes; needed for TPUs.
    images, labels = tf.data.make_one_shot_iterator(dataset).get_next()
    # return images, labels
    images.set_shape([batch_size, 28, 28, 1])
    labels.set_shape([
        batch_size,
    ])
    return images, labels

  return input_fn
