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
"""Tests for tensorflow_privacy.privacy.membership_inference_attack.data_structures."""
import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import _log_value
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import RocCurve
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SingleSliceSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingFeature


class SingleSliceSpecTest(parameterized.TestCase):

  def testStrEntireDataset(self):
    self.assertEqual(str(SingleSliceSpec()), 'Entire dataset')

  @parameterized.parameters(
      (SlicingFeature.CLASS, 2, 'CLASS=2'),
      (SlicingFeature.PERCENTILE, (10, 20), 'Loss percentiles: 10-20'),
      (SlicingFeature.CORRECTLY_CLASSIFIED, True, 'CORRECTLY_CLASSIFIED=True'),
  )
  def testStr(self, feature, value, expected_str):
    self.assertEqual(str(SingleSliceSpec(feature, value)), expected_str)


class AttackInputDataTest(absltest.TestCase):

  def test_get_loss_from_logits(self):
    attack_input = AttackInputData(
        logits_train=np.array([[-0.3, 1.5, 0.2], [2, 3, 0.5]]),
        logits_test=np.array([[2, 0.3, 0.2], [0.3, -0.5, 0.2]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0, 2]))

    np.testing.assert_allclose(
        attack_input.get_loss_train(), [0.36313551, 1.37153903], atol=1e-7)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), [0.29860897, 0.95618669], atol=1e-7)

  def test_get_loss_from_probs(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.1, 0.1, 0.8], [0.8, 0.2, 0]]),
        probs_test=np.array([[0, 0.0001, 0.9999], [0.07, 0.18, 0.75]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0, 2]))

    np.testing.assert_allclose(
        attack_input.get_loss_train(), [2.30258509, 0.2231436], atol=1e-7)
    np.testing.assert_allclose(
        attack_input.get_loss_test(), [18.42068074, 0.28768207], atol=1e-7)

  def test_get_loss_explicitly_provided(self):
    attack_input = AttackInputData(
        loss_train=np.array([1.0, 3.0, 6.0]),
        loss_test=np.array([1.0, 4.0, 6.0]))

    np.testing.assert_equal(attack_input.get_loss_train().tolist(),
                            [1.0, 3.0, 6.0])
    np.testing.assert_equal(attack_input.get_loss_test().tolist(),
                            [1.0, 4.0, 6.0])

  def test_get_probs_sizes(self):
    attack_input = AttackInputData(
        probs_train=np.array([[0.1, 0.1, 0.8], [0.8, 0.2, 0]]),
        probs_test=np.array([[0, 0.0001, 0.9999]]),
        labels_train=np.array([1, 0]),
        labels_test=np.array([0]))

    np.testing.assert_equal(attack_input.get_train_size(), 2)
    np.testing.assert_equal(attack_input.get_test_size(), 1)

  def test_get_entropy(self):
    attack_input = AttackInputData(
        logits_train=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        logits_test=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        labels_train=np.array([0, 2]),
        labels_test=np.array([0, 2]))

    np.testing.assert_equal(attack_input.get_entropy_train().tolist(), [0, 0])
    np.testing.assert_equal(attack_input.get_entropy_test().tolist(),
                            [2 * _log_value(0), 0])

    attack_input = AttackInputData(
        logits_train=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        logits_test=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    np.testing.assert_equal(attack_input.get_entropy_train().tolist(), [0, 0])
    np.testing.assert_equal(attack_input.get_entropy_test().tolist(), [0, 0])

  def test_get_entropy_explicitly_provided(self):
    attack_input = AttackInputData(
        entropy_train=np.array([0.0, 2.0, 1.0]),
        entropy_test=np.array([0.5, 3.0, 5.0]))

    np.testing.assert_equal(attack_input.get_entropy_train().tolist(),
                            [0.0, 2.0, 1.0])
    np.testing.assert_equal(attack_input.get_entropy_test().tolist(),
                            [0.5, 3.0, 5.0])

  def test_validator(self):
    self.assertRaises(ValueError,
                      AttackInputData(logits_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(probs_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(labels_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(loss_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(entropy_train=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(logits_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(probs_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(labels_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(loss_test=np.array([])).validate)
    self.assertRaises(ValueError,
                      AttackInputData(entropy_test=np.array([])).validate)
    self.assertRaises(ValueError, AttackInputData().validate)
    # Tests that having both logits and probs are not allowed.
    self.assertRaises(
        ValueError,
        AttackInputData(
            logits_train=np.array([]),
            logits_test=np.array([]),
            probs_train=np.array([]),
            probs_test=np.array([])).validate)


class RocCurveTest(absltest.TestCase):

  def test_auc_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_auc(), 0.5)

  def test_auc_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_auc(), 1.0)

  def test_attacker_advantage_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_attacker_advantage(), 0.0)

  def test_attacker_advantage_perfect_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 1.0, 1.0]),
        fpr=np.array([1.0, 1.0, 0.0]),
        thresholds=np.array([0, 1, 2]))

    self.assertEqual(roc.get_auc(), 1.0)


class SingleAttackResultTest(absltest.TestCase):

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_auc_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.get_auc(), 0.5)

  # Only a basic test, as this method calls RocCurve which is tested separately.
  def test_attacker_advantage_random_classifier(self):
    roc = RocCurve(
        tpr=np.array([0.0, 0.5, 1.0]),
        fpr=np.array([0.0, 0.5, 1.0]),
        thresholds=np.array([0, 1, 2]))

    result = SingleAttackResult(
        roc_curve=roc,
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK)

    self.assertEqual(result.get_attacker_advantage(), 0.0)


class AttackResultsCollectionTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(AttackResultsCollectionTest, self).__init__(*args, **kwargs)

    self.some_attack_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 0.5, 1.0]),
            fpr=np.array([0.0, 0.5, 1.0]),
            thresholds=np.array([0, 1, 2])))

    self.results_epoch_10 = AttackResults(
        single_attack_results=[self.some_attack_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.4,
            accuracy_test=0.3,
            epoch_num=10,
            model_variant_label='default'))

    self.results_epoch_15 = AttackResults(
        single_attack_results=[self.some_attack_result],
        privacy_report_metadata=PrivacyReportMetadata(
            accuracy_train=0.5,
            accuracy_test=0.4,
            epoch_num=15,
            model_variant_label='default'))

    self.attack_results_no_metadata = AttackResults(
        single_attack_results=[self.some_attack_result])

    self.collection_with_metadata = AttackResultsCollection(
        [self.results_epoch_10, self.results_epoch_15])

    self.collection_no_metadata = AttackResultsCollection(
        [self.attack_results_no_metadata, self.attack_results_no_metadata])

  def test_save_with_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
      self.collection_with_metadata.save(tmpdirname)
      loaded_collection = AttackResultsCollection.load(tmpdirname)

    self.assertEqual(
        repr(self.collection_with_metadata), repr(loaded_collection))
    self.assertLen(loaded_collection.attack_results_list, 2)

  def test_save_no_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
      self.collection_no_metadata.save(tmpdirname)
      loaded_collection = AttackResultsCollection.load(tmpdirname)

    self.assertEqual(repr(self.collection_no_metadata), repr(loaded_collection))
    self.assertLen(loaded_collection.attack_results_list, 2)


class AttackResultsTest(absltest.TestCase):

  perfect_classifier_result: SingleAttackResult
  random_classifier_result: SingleAttackResult

  def __init__(self, *args, **kwargs):
    super(AttackResultsTest, self).__init__(*args, **kwargs)

    # ROC curve of a perfect classifier
    self.perfect_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(SlicingFeature.CORRECTLY_CLASSIFIED, True),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 1.0, 1.0]),
            fpr=np.array([1.0, 1.0, 0.0]),
            thresholds=np.array([0, 1, 2])))

    # ROC curve of a random classifier
    self.random_classifier_result = SingleAttackResult(
        slice_spec=SingleSliceSpec(None),
        attack_type=AttackType.THRESHOLD_ATTACK,
        roc_curve=RocCurve(
            tpr=np.array([0.0, 0.5, 1.0]),
            fpr=np.array([0.0, 0.5, 1.0]),
            thresholds=np.array([0, 1, 2])))

  def test_get_result_with_max_auc_first(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(results.get_result_with_max_auc(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_auc_second(self):
    results = AttackResults(
        [self.random_classifier_result, self.perfect_classifier_result])
    self.assertEqual(results.get_result_with_max_auc(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_attacker_advantage_first(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(results.get_result_with_max_attacker_advantage(),
                     self.perfect_classifier_result)

  def test_get_result_with_max_attacker_advantage_second(self):
    results = AttackResults(
        [self.random_classifier_result, self.perfect_classifier_result])
    self.assertEqual(results.get_result_with_max_attacker_advantage(),
                     self.perfect_classifier_result)

  def test_summary_by_slices(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(
        results.summary(by_slices=True),
        'Best-performing attacks over all slices\n' +
        '  THRESHOLD_ATTACK achieved an AUC of 1.00 ' +
        'on slice CORRECTLY_CLASSIFIED=True\n' +
        '  THRESHOLD_ATTACK achieved an advantage of 1.00 ' +
        'on slice CORRECTLY_CLASSIFIED=True\n\n' +
        'Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=True"\n' +
        '  THRESHOLD_ATTACK achieved an AUC of 1.00\n' +
        '  THRESHOLD_ATTACK achieved an advantage of 1.00\n\n' +
        'Best-performing attacks over slice: "Entire dataset"\n' +
        '  THRESHOLD_ATTACK achieved an AUC of 0.50\n' +
        '  THRESHOLD_ATTACK achieved an advantage of 0.00')

  def test_summary_without_slices(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])
    self.assertEqual(
        results.summary(by_slices=False),
        'Best-performing attacks over all slices\n' +
        '  THRESHOLD_ATTACK achieved an AUC of 1.00 ' +
        'on slice CORRECTLY_CLASSIFIED=True\n' +
        '  THRESHOLD_ATTACK achieved an advantage of 1.00 ' +
        'on slice CORRECTLY_CLASSIFIED=True')

  def test_save_load(self):
    results = AttackResults(
        [self.perfect_classifier_result, self.random_classifier_result])

    with tempfile.TemporaryDirectory() as tmpdirname:
      filepath = os.path.join(tmpdirname, 'results.pickle')
      results.save(filepath)
      loaded_results = AttackResults.load(filepath)

    self.assertEqual(repr(results), repr(loaded_results))

  def test_calculate_pd_dataframe(self):
    single_results = [
        self.perfect_classifier_result, self.random_classifier_result
    ]
    results = AttackResults(single_results)
    df = results.calculate_pd_dataframe()
    df_expected = pd.DataFrame({
        'slice feature': ['correctly_classified', 'Entire dataset'],
        'slice value': ['True', ''],
        'attack type': ['THRESHOLD_ATTACK', 'THRESHOLD_ATTACK'],
        'Attacker advantage': [1.0, 0.0],
        'AUC': [1.0, 0.5]
    })
    pd.testing.assert_frame_equal(df, df_expected)


if __name__ == '__main__':
  absltest.main()
