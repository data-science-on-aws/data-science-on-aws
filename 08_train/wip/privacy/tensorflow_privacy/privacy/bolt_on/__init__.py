# Copyright 2019, The TensorFlow Privacy Authors.
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
"""BoltOn Method for privacy."""
import sys
from distutils.version import LooseVersion
import tensorflow.compat.v1 as tf

if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
  raise ImportError("Please upgrade your version "
                    "of tensorflow from: {0} to at least 2.0.0 to "
                    "use privacy/bolt_on".format(LooseVersion(tf.__version__)))
if hasattr(sys, "skip_tf_privacy_import"):  # Useful for standalone scripts.
  pass
else:
  from tensorflow_privacy.privacy.bolt_on.models import BoltOnModel  # pylint: disable=g-import-not-at-top
  from tensorflow_privacy.privacy.bolt_on.optimizers import BoltOn  # pylint: disable=g-import-not-at-top
  from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexHuber  # pylint: disable=g-import-not-at-top
  from tensorflow_privacy.privacy.bolt_on.losses import StrongConvexBinaryCrossentropy  # pylint: disable=g-import-not-at-top
