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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib


class ComputeDpSgdPrivacyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Test0', 60000, 150, 1.3, 15, 1e-5, 0.941870567, 19.0),
      ('Test1', 100000, 100, 1.0, 30, 1e-7, 1.70928734, 13.0),
      ('Test2', 100000000, 1024, 0.1, 10, 1e-7, 5907984.81339406, 1.25),
  )
  def test_compute_dp_sgd_privacy(self, n, batch_size, noise_multiplier, epochs,
                                  delta, expected_eps, expected_order):
    eps, order = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        n, batch_size, noise_multiplier, epochs, delta)
    self.assertAlmostEqual(eps, expected_eps)
    self.assertAlmostEqual(order, expected_order)

if __name__ == '__main__':
  absltest.main()
