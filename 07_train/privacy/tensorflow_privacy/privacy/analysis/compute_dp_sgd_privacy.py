# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Command-line script for computing privacy of a model trained with DP-SGD.

The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism. The mechanism's parameters are controlled by flags.

Example:
  compute_dp_sgd_privacy
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5

The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags

from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

# Opting out of loading all sibling packages and their dependencies.
sys.skip_tf_privacy_import = True

FLAGS = flags.FLAGS

flags.DEFINE_integer('N', None, 'Total number of examples')
flags.DEFINE_integer('batch_size', None, 'Batch size')
flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')
flags.DEFINE_float('epochs', None, 'Number of epochs (may be fractional)')
flags.DEFINE_float('delta', 1e-6, 'Target delta')


def main(argv):
  del argv  # argv is not used.

  assert FLAGS.N is not None, 'Flag N is missing.'
  assert FLAGS.batch_size is not None, 'Flag batch_size is missing.'
  assert FLAGS.noise_multiplier is not None, 'Flag noise_multiplier is missing.'
  assert FLAGS.epochs is not None, 'Flag epochs is missing.'
  compute_dp_sgd_privacy(FLAGS.N, FLAGS.batch_size, FLAGS.noise_multiplier,
                         FLAGS.epochs, FLAGS.delta)


if __name__ == '__main__':
  app.run(main)
