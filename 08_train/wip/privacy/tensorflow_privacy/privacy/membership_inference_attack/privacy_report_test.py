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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.privacy_report."""
from absl.testing import absltest
import numpy as np

from tensorflow_privacy.privacy.membership_inference_attack import privacy_report

from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import \
  PrivacyReportMetadata
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import RocCurve
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec


class PrivacyReportTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(PrivacyReportTest, self).__init__(*args, **kwargs)

    # Classifier that achieves an AUC of 0.5.
    self.imperfect_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 0.5, 1.0]),
            fpr=np.array([0.0, 0.5, 1.0]),
            thresholds=np.array([0, 1, 2])))

    # Classifier that achieves an AUC of 1.0.
    self.perfect_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 1.0, 1.0]),
            fpr=np.array([1.0, 1.0, 0.0]),
            thresholds=np.array([0, 1, 2])))

    self.results_epoch_10 = AttackResults(
        single_attack_results=[self.imperfect_classifier_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.4,
            accuracy_test=0.3,
            epoch_num=10,
            model_variant_label='default'))

    self.results_epoch_15 = AttackResults(
        single_attack_results=[self.perfect_classifier_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.5,
            accuracy_test=0.4,
            epoch_num=15,
            model_variant_label='default'))

    self.results_epoch_15_model_2 = AttackResults(
        single_attack_results=[self.perfect_classifier_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.6,
            accuracy_test=0.7,
            epoch_num=15,
            model_variant_label='model 2'))

    self.attack_results_no_metadata = AttackResults(
        single_attack_results=[self.perfect_classifier_result])

  def test_plot_by_epochs_no_metadata(self):
    # Raise error if metadata is missing
    self.assertRaises(
        ValueError, privacy_report.plot_by_epochs,
        AttackResultsCollection((self.attack_results_no_metadata,)), ['AUC'])

  def test_single_metric_plot_by_epochs(self):
    fig = privacy_report.plot_by_epochs(
        AttackResultsCollection((self.results_epoch_10, self.results_epoch_15)),
        ['AUC'])
    # extract data from figure.
    auc_data = fig.gca().lines[0].get_data()
    # X axis lists epoch values
    np.testing.assert_array_equal(auc_data[0], [10, 15])
    # Y axis lists AUC values
    np.testing.assert_array_equal(auc_data[1], [0.5, 1.0])
    # Check the title
    self.assertEqual(fig._suptitle.get_text(), 'Vulnerability per Epoch')

  def test_multiple_metrics_plot_by_epochs(self):
    fig = privacy_report.plot_by_epochs(
        AttackResultsCollection((self.results_epoch_10, self.results_epoch_15)),
        ['AUC', 'Attacker advantage'])
    # extract data from figure.
    auc_data = fig.axes[0].lines[0].get_data()
    attacker_advantage_data = fig.axes[1].lines[0].get_data()
    # X axis lists epoch values
    np.testing.assert_array_equal(auc_data[0], [10, 15])
    np.testing.assert_array_equal(attacker_advantage_data[0], [10, 15])
    # Y axis lists privacy metrics
    np.testing.assert_array_equal(auc_data[1], [0.5, 1.0])
    np.testing.assert_array_equal(attacker_advantage_data[1], [0, 1.0])
    # Check the title
    self.assertEqual(fig._suptitle.get_text(), 'Vulnerability per Epoch')

  def test_multiple_metrics_plot_by_epochs_multiple_models(self):
    fig = privacy_report.plot_by_epochs(
        AttackResultsCollection((self.results_epoch_10, self.results_epoch_15,
                                 self.results_epoch_15_model_2)),
        ['AUC', 'Attacker advantage'])
    # extract data from figure.
    # extract data from figure.
    auc_data_model_1 = fig.axes[0].lines[0].get_data()
    auc_data_model_2 = fig.axes[0].lines[1].get_data()
    attacker_advantage_data_model_1 = fig.axes[1].lines[0].get_data()
    attacker_advantage_data_model_2 = fig.axes[1].lines[1].get_data()
    # X axis lists epoch values
    np.testing.assert_array_equal(auc_data_model_1[0], [10, 15])
    np.testing.assert_array_equal(auc_data_model_2[0], [15])
    np.testing.assert_array_equal(attacker_advantage_data_model_1[0], [10, 15])
    np.testing.assert_array_equal(attacker_advantage_data_model_2[0], [15])
    # Y axis lists privacy metrics
    np.testing.assert_array_equal(auc_data_model_1[1], [0.5, 1.0])
    np.testing.assert_array_equal(auc_data_model_2[1], [1.0])
    np.testing.assert_array_equal(attacker_advantage_data_model_1[1], [0, 1.0])
    np.testing.assert_array_equal(attacker_advantage_data_model_2[1], [1.0])
    # Check the title
    self.assertEqual(fig._suptitle.get_text(), 'Vulnerability per Epoch')

  def test_plot_privacy_vs_accuracy_single_model_no_metadata(self):
    # Raise error if metadata is missing
    self.assertRaises(
        ValueError, privacy_report.plot_privacy_vs_accuracy,
        AttackResultsCollection((self.attack_results_no_metadata,)), ['AUC'])

  def test_single_metric_plot_privacy_vs_accuracy_single_model(self):
    fig = privacy_report.plot_privacy_vs_accuracy(
        AttackResultsCollection((self.results_epoch_10, self.results_epoch_15)),
        ['AUC'])
    # extract data from figure.
    auc_data = fig.gca().lines[0].get_data()
    # X axis lists epoch values
    np.testing.assert_array_equal(auc_data[0], [0.4, 0.5])
    # Y axis lists AUC values
    np.testing.assert_array_equal(auc_data[1], [0.5, 1.0])
    # Check the title
    self.assertEqual(fig._suptitle.get_text(), 'Privacy vs Utility Analysis')

  def test_multiple_metrics_plot_privacy_vs_accuracy_single_model(self):
    fig = privacy_report.plot_privacy_vs_accuracy(
        AttackResultsCollection((self.results_epoch_10, self.results_epoch_15)),
        ['AUC', 'Attacker advantage'])
    # extract data from figure.
    auc_data = fig.axes[0].lines[0].get_data()
    attacker_advantage_data = fig.axes[1].lines[0].get_data()
    # X axis lists epoch values
    np.testing.assert_array_equal(auc_data[0], [0.4, 0.5])
    np.testing.assert_array_equal(attacker_advantage_data[0], [0.4, 0.5])
    # Y axis lists privacy metrics
    np.testing.assert_array_equal(auc_data[1], [0.5, 1.0])
    np.testing.assert_array_equal(attacker_advantage_data[1], [0, 1.0])
    # Check the title
    self.assertEqual(fig._suptitle.get_text(), 'Privacy vs Utility Analysis')

  def test_multiple_metrics_plot_privacy_vs_accuracy_multiple_model(self):
    fig = privacy_report.plot_privacy_vs_accuracy(
        AttackResultsCollection((self.results_epoch_10, self.results_epoch_15,
                                 self.results_epoch_15_model_2)),
        ['AUC', 'Attacker advantage'])
    # extract data from figure.
    auc_data_model_1 = fig.axes[0].lines[0].get_data()
    auc_data_model_2 = fig.axes[0].lines[1].get_data()
    attacker_advantage_data_model_1 = fig.axes[1].lines[0].get_data()
    attacker_advantage_data_model_2 = fig.axes[1].lines[1].get_data()
    # X axis lists epoch values
    np.testing.assert_array_equal(auc_data_model_1[0], [0.4, 0.5])
    np.testing.assert_array_equal(auc_data_model_2[0], [0.6])
    np.testing.assert_array_equal(attacker_advantage_data_model_1[0],
                                  [0.4, 0.5])
    np.testing.assert_array_equal(attacker_advantage_data_model_2[0], [0.6])
    # Y axis lists privacy metrics
    np.testing.assert_array_equal(auc_data_model_1[1], [0.5, 1.0])
    np.testing.assert_array_equal(auc_data_model_2[1], [1.0])
    np.testing.assert_array_equal(attacker_advantage_data_model_1[1], [0, 1.0])
    np.testing.assert_array_equal(attacker_advantage_data_model_2[1], [1.0])
    # Check the title
    self.assertEqual(fig._suptitle.get_text(), 'Privacy vs Utility Analysis')


if __name__ == '__main__':
  absltest.main()
