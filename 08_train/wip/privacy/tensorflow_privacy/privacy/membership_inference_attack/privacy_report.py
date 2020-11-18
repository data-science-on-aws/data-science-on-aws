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
"""Plotting code for ML Privacy Reports."""
from typing import Iterable
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResultsDFColumns
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import ENTIRE_DATASET_SLICE_STR
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import PrivacyMetric

# Helper constants for DataFrame keys.
LEGEND_LABEL_STR = 'legend label'
EPOCH_STR = 'Epoch'
TRAIN_ACCURACY_STR = 'Train accuracy'


def plot_by_epochs(results: AttackResultsCollection,
                   privacy_metrics: Iterable[PrivacyMetric]) -> plt.Figure:
  """Plots privacy vulnerabilities vs epoch numbers.

  In case multiple privacy metrics are specified, the plot will feature
  multiple subplots (one subplot per metrics). Multiple model variants
  are supported.
  Args:
    results: AttackResults for the plot
    privacy_metrics: List of enumerated privacy metrics that should be plotted.

  Returns:
    A pyplot figure with privacy vs accuracy plots.
  """

  _validate_results(results.attack_results_list)
  all_results_df = _calculate_combined_df_with_metadata(
      results.attack_results_list)
  return _generate_subplots(
      all_results_df=all_results_df,
      x_axis_metric='Epoch',
      figure_title='Vulnerability per Epoch',
      privacy_metrics=privacy_metrics)


def plot_privacy_vs_accuracy(results: AttackResultsCollection,
                             privacy_metrics: Iterable[PrivacyMetric]):
  """Plots privacy vulnerabilities vs accuracy plots.

  In case multiple privacy metrics are specified, the plot will feature
  multiple subplots (one subplot per metrics). Multiple model variants
  are supported.
  Args:
    results: AttackResults for the plot
    privacy_metrics: List of enumerated privacy metrics that should be plotted.

  Returns:
    A pyplot figure with privacy vs accuracy plots.

  """
  _validate_results(results.attack_results_list)
  all_results_df = _calculate_combined_df_with_metadata(
      results.attack_results_list)
  return _generate_subplots(
      all_results_df=all_results_df,
      x_axis_metric='Train accuracy',
      figure_title='Privacy vs Utility Analysis',
      privacy_metrics=privacy_metrics)


def _calculate_combined_df_with_metadata(results: Iterable[AttackResults]):
  """Adds metadata to the dataframe and concats them together."""
  all_results_df = None
  for attack_results in results:
    attack_results_df = attack_results.calculate_pd_dataframe()
    attack_results_df = attack_results_df.loc[attack_results_df[str(
        AttackResultsDFColumns.SLICE_FEATURE)] == ENTIRE_DATASET_SLICE_STR]
    attack_results_df.insert(0, EPOCH_STR,
                             attack_results.privacy_report_metadata.epoch_num)
    attack_results_df.insert(
        0, TRAIN_ACCURACY_STR,
        attack_results.privacy_report_metadata.accuracy_train)
    attack_results_df.insert(
        0, LEGEND_LABEL_STR,
        attack_results.privacy_report_metadata.model_variant_label + ' - ' +
        attack_results_df[str(AttackResultsDFColumns.ATTACK_TYPE)])
    if all_results_df is None:
      all_results_df = attack_results_df
    else:
      all_results_df = pd.concat([all_results_df, attack_results_df],
                                 ignore_index=True)
  return all_results_df


def _generate_subplots(all_results_df: pd.DataFrame, x_axis_metric: str,
                       figure_title: str,
                       privacy_metrics: Iterable[PrivacyMetric]):
  """Create one subplot per privacy metric for a specified x_axis_metric."""
  fig, axes = plt.subplots(
      1, len(privacy_metrics), figsize=(5 * len(privacy_metrics) + 3, 5))
  # Set a title for the entire group of subplots.
  fig.suptitle(figure_title)
  if len(privacy_metrics) == 1:
    axes = (axes,)
  for i, privacy_metric in enumerate(privacy_metrics):
    legend_labels = all_results_df[LEGEND_LABEL_STR].unique()
    for legend_label in legend_labels:
      single_label_results = all_results_df.loc[all_results_df[LEGEND_LABEL_STR]
                                                == legend_label]
      sorted_label_results = single_label_results.sort_values(x_axis_metric)
      axes[i].plot(sorted_label_results[x_axis_metric],
                   sorted_label_results[str(privacy_metric)])
    axes[i].set_xlabel(x_axis_metric)
    axes[i].set_title('%s for %s' % (privacy_metric, ENTIRE_DATASET_SLICE_STR))
  plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1))
  fig.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for suptitle.

  return fig


def _validate_results(results: Iterable[AttackResults]):
  for attack_results in results:
    if not attack_results or not attack_results.privacy_report_metadata:
      raise ValueError('Privacy metadata is not defined.')
    if not attack_results.privacy_report_metadata.epoch_num:
      raise ValueError('epoch_num in metadata is not defined.')
