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
"""Specifying and creating AttackInputData slices."""

import collections
import copy
from typing import List

import numpy as np
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingFeature
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec


def _slice_if_not_none(a, idx):
  return None if a is None else a[idx]


def _slice_data_by_indices(data: AttackInputData, idx_train,
                           idx_test) -> AttackInputData:
  """Slices train fields with with idx_train and test fields with and idx_test."""

  result = AttackInputData()

  # Slice train data.
  result.logits_train = _slice_if_not_none(data.logits_train, idx_train)
  result.probs_train = _slice_if_not_none(data.probs_train, idx_train)
  result.labels_train = _slice_if_not_none(data.labels_train, idx_train)
  result.loss_train = _slice_if_not_none(data.loss_train, idx_train)
  result.entropy_train = _slice_if_not_none(data.entropy_train, idx_train)

  # Slice test data.
  result.logits_test = _slice_if_not_none(data.logits_test, idx_test)
  result.probs_test = _slice_if_not_none(data.probs_test, idx_test)
  result.labels_test = _slice_if_not_none(data.labels_test, idx_test)
  result.loss_test = _slice_if_not_none(data.loss_test, idx_test)
  result.entropy_test = _slice_if_not_none(data.entropy_test, idx_test)

  return result


def _slice_by_class(data: AttackInputData, class_value: int) -> AttackInputData:
  idx_train = data.labels_train == class_value
  idx_test = data.labels_test == class_value
  return _slice_data_by_indices(data, idx_train, idx_test)


def _slice_by_percentiles(data: AttackInputData, from_percentile: float,
                          to_percentile: float):
  """Slices samples by loss percentiles."""

  # Find from_percentile and to_percentile percentiles in losses.
  loss_train = data.get_loss_train()
  loss_test = data.get_loss_test()
  losses = np.concatenate((loss_train, loss_test))
  from_loss = np.percentile(losses, from_percentile)
  to_loss = np.percentile(losses, to_percentile)

  idx_train = (from_loss <= loss_train) & (loss_train <= to_loss)
  idx_test = (from_loss <= loss_test) & (loss_test <= to_loss)

  return _slice_data_by_indices(data, idx_train, idx_test)


def _indices_by_classification(logits_or_probs, labels, correctly_classified):
  idx_correct = labels == np.argmax(logits_or_probs, axis=1)
  return idx_correct if correctly_classified else np.invert(idx_correct)


def _slice_by_classification_correctness(data: AttackInputData,
                                         correctly_classified: bool):
  idx_train = _indices_by_classification(data.logits_or_probs_train,
                                         data.labels_train,
                                         correctly_classified)
  idx_test = _indices_by_classification(data.logits_or_probs_test,
                                        data.labels_test, correctly_classified)
  return _slice_data_by_indices(data, idx_train, idx_test)


def get_single_slice_specs(slicing_spec: SlicingSpec,
                           num_classes: int = None) -> List[SingleSliceSpec]:
  """Returns slices of data according to slicing_spec."""
  result = []

  if slicing_spec.entire_dataset:
    result.append(SingleSliceSpec())

  # Create slices by class.
  by_class = slicing_spec.by_class
  if isinstance(by_class, bool):
    if by_class:
      assert num_classes, "When by_class == True, num_classes should be given."
      assert 0 <= num_classes <= 1000, (
          f"Too much classes for slicing by classes. "
          f"Found {num_classes}.")
      for c in range(num_classes):
        result.append(SingleSliceSpec(SlicingFeature.CLASS, c))
  elif isinstance(by_class, int):
    result.append(SingleSliceSpec(SlicingFeature.CLASS, by_class))
  elif isinstance(by_class, collections.Iterable):
    for c in by_class:
      result.append(SingleSliceSpec(SlicingFeature.CLASS, c))

  # Create slices by percentiles
  if slicing_spec.by_percentiles:
    for percent in range(0, 100, 10):
      result.append(
          SingleSliceSpec(SlicingFeature.PERCENTILE, (percent, percent + 10)))

  # Create slices by correctness of the classifications.
  if slicing_spec.by_classification_correctness:
    result.append(SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, True))
    result.append(SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, False))

  return result


def get_slice(data: AttackInputData,
              slice_spec: SingleSliceSpec) -> AttackInputData:
  """Returns a single slice of data according to slice_spec."""
  if slice_spec.entire_dataset:
    data_slice = copy.copy(data)
  elif slice_spec.feature == SlicingFeature.CLASS:
    data_slice = _slice_by_class(data, slice_spec.value)
  elif slice_spec.feature == SlicingFeature.PERCENTILE:
    from_percentile, to_percentile = slice_spec.value
    data_slice = _slice_by_percentiles(data, from_percentile, to_percentile)
  elif slice_spec.feature == SlicingFeature.CORRECTLY_CLASSIFIED:
    data_slice = _slice_by_classification_correctness(data, slice_spec.value)
  else:
    raise ValueError('Unknown slice spec feature "%s"' % slice_spec.feature)

  data_slice.slice_spec = slice_spec
  return data_slice
