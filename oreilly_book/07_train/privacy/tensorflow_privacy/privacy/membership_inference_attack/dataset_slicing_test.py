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

# Lint as: python3
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.dataset_slicing."""

from absl.testing import absltest
import numpy as np

from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingFeature
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.dataset_slicing import get_single_slice_specs
from tensorflow_privacy.privacy.membership_inference_attack.dataset_slicing import get_slice


def _are_all_fields_equal(lhs, rhs) -> bool:
  return vars(lhs) == vars(rhs)


def _are_lists_equal(lhs, rhs) -> bool:
  if len(lhs) != len(rhs):
    return False
  for l, r in zip(lhs, rhs):
    if not _are_all_fields_equal(l, r):
      return False
  return True


class SingleSliceSpecsTest(absltest.TestCase):
  """Tests for get_single_slice_specs."""

  ENTIRE_DATASET_SLICE = SingleSliceSpec()

  def test_no_slices(self):
    input_data = SlicingSpec(entire_dataset=False)
    expected = []
    output = get_single_slice_specs(input_data)
    self.assertTrue(_are_lists_equal(output, expected))

  def test_entire_dataset(self):
    input_data = SlicingSpec()
    expected = [self.ENTIRE_DATASET_SLICE]
    output = get_single_slice_specs(input_data)
    self.assertTrue(_are_lists_equal(output, expected))

  def test_slice_by_classes(self):
    input_data = SlicingSpec(by_class=True)
    n_classes = 5
    expected = [self.ENTIRE_DATASET_SLICE] + [
        SingleSliceSpec(SlicingFeature.CLASS, c) for c in range(n_classes)
    ]
    output = get_single_slice_specs(input_data, n_classes)
    self.assertTrue(_are_lists_equal(output, expected))

  def test_slice_by_percentiles(self):
    input_data = SlicingSpec(entire_dataset=False, by_percentiles=True)
    expected0 = SingleSliceSpec(SlicingFeature.PERCENTILE, (0, 10))
    expected5 = SingleSliceSpec(SlicingFeature.PERCENTILE, (50, 60))
    output = get_single_slice_specs(input_data)
    self.assertLen(output, 10)
    self.assertTrue(_are_all_fields_equal(output[0], expected0))
    self.assertTrue(_are_all_fields_equal(output[5], expected5))

  def test_slice_by_correcness(self):
    input_data = SlicingSpec(
        entire_dataset=False, by_classification_correctness=True)
    expected = SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, True)
    output = get_single_slice_specs(input_data)
    self.assertLen(output, 2)
    self.assertTrue(_are_all_fields_equal(output[0], expected))

  def test_slicing_by_multiple_features(self):
    input_data = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=True,
        by_classification_correctness=True)
    n_classes = 10
    expected_slices = n_classes
    expected_slices += 1  # entire dataset slice
    expected_slices += 10  # percentiles slices
    expected_slices += 2  # correcness classification slices
    output = get_single_slice_specs(input_data, n_classes)
    self.assertLen(output, expected_slices)


class GetSliceTest(absltest.TestCase):

  def __init__(self, methodname):
    """Initialize the test class."""
    super().__init__(methodname)

    # Create test data for 3 class classification task.
    logits_train = np.array([[0, 1, 0], [2, 0, 3], [4, 5, 0], [6, 7, 0]])
    logits_test = np.array([[10, 0, 11], [12, 13, 0], [14, 15, 0], [0, 16, 17]])
    probs_train = np.array([[0, 1, 0], [0.1, 0, 0.7], [0.4, 0.6, 0],
                            [0.3, 0.7, 0]])
    probs_test = np.array([[0.4, 0, 0.6], [0.1, 0.9, 0], [0.15, 0.85, 0],
                           [0, 0, 1]])
    labels_train = np.array([1, 0, 1, 2])
    labels_test = np.array([1, 2, 0, 2])
    loss_train = np.array([2, 0.25, 4, 3])
    loss_test = np.array([0.5, 3.5, 7, 4.5])
    entropy_train = np.array([0.4, 8, 0.6, 10])
    entropy_test = np.array([15, 10.5, 4.5, 0.3])

    self.input_data = AttackInputData(
        logits_train=logits_train,
        logits_test=logits_test,
        probs_train=probs_train,
        probs_test=probs_test,
        labels_train=labels_train,
        labels_test=labels_test,
        loss_train=loss_train,
        loss_test=loss_test,
        entropy_train=entropy_train,
        entropy_test=entropy_test)

  def test_slice_entire_dataset(self):
    entire_dataset_slice = SingleSliceSpec()
    output = get_slice(self.input_data, entire_dataset_slice)
    expected = self.input_data
    expected.slice_spec = entire_dataset_slice
    self.assertTrue(_are_all_fields_equal(output, self.input_data))

  def test_slice_by_class(self):
    class_index = 1
    class_slice = SingleSliceSpec(SlicingFeature.CLASS, class_index)
    output = get_slice(self.input_data, class_slice)

    # Check logits.
    self.assertLen(output.logits_train, 2)
    self.assertLen(output.logits_test, 1)
    self.assertTrue((output.logits_train[1] == [4, 5, 0]).all())

    # Check probs.
    self.assertLen(output.probs_train, 2)
    self.assertLen(output.probs_test, 1)
    self.assertTrue((output.probs_train[1] == [0.4, 0.6, 0]).all())

    # Check labels.
    self.assertLen(output.labels_train, 2)
    self.assertLen(output.labels_test, 1)
    self.assertTrue((output.labels_train == class_index).all())
    self.assertTrue((output.labels_test == class_index).all())

    # Check losses
    self.assertLen(output.loss_train, 2)
    self.assertLen(output.loss_test, 1)
    self.assertTrue((output.loss_train == [2, 4]).all())
    self.assertTrue((output.loss_test == [0.5]).all())

    # Check entropy
    self.assertLen(output.entropy_train, 2)
    self.assertLen(output.entropy_test, 1)
    self.assertTrue((output.entropy_train == [0.4, 0.6]).all())
    self.assertTrue((output.entropy_test == [15]).all())

  def test_slice_by_percentile(self):
    percentile_slice = SingleSliceSpec(SlicingFeature.PERCENTILE, (0, 50))
    output = get_slice(self.input_data, percentile_slice)

    # Check logits.
    self.assertLen(output.logits_train, 3)
    self.assertLen(output.logits_test, 1)
    self.assertTrue((output.logits_test[0] == [10, 0, 11]).all())

    # Check labels.
    self.assertLen(output.labels_train, 3)
    self.assertLen(output.labels_test, 1)
    self.assertTrue((output.labels_train == [1, 0, 2]).all())
    self.assertTrue((output.labels_test == [1]).all())

  def test_slice_by_correctness(self):
    percentile_slice = SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED,
                                       False)
    output = get_slice(self.input_data, percentile_slice)

    # Check logits.
    self.assertLen(output.logits_train, 2)
    self.assertLen(output.logits_test, 3)
    self.assertTrue((output.logits_train[1] == [6, 7, 0]).all())
    self.assertTrue((output.logits_test[1] == [12, 13, 0]).all())

    # Check labels.
    self.assertLen(output.labels_train, 2)
    self.assertLen(output.labels_test, 3)
    self.assertTrue((output.labels_train == [0, 2]).all())
    self.assertTrue((output.labels_test == [1, 2, 0]).all())


if __name__ == '__main__':
  absltest.main()
