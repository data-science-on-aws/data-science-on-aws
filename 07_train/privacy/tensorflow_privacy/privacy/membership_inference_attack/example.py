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
"""An example for the membership inference attacks.

This is using a toy model based on classifying four spacial clusters of data.
"""
import os
import tempfile

from absl import app
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia

from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResults
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import \
  PrivacyReportMetadata
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting
import tensorflow_privacy.privacy.membership_inference_attack.privacy_report as privacy_report


def generate_random_cluster(center, scale, num_points):
  return np.random.normal(size=(num_points, len(center))) * scale + center


def generate_features_and_labels(samples_per_cluster=250, scale=0.1):
  """Generates noised 3D clusters."""
  cluster_centers = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]

  features = np.concatenate((
      generate_random_cluster(
          center=cluster_centers[0],
          scale=scale,
          num_points=samples_per_cluster),
      generate_random_cluster(
          center=cluster_centers[1],
          scale=scale,
          num_points=samples_per_cluster),
      generate_random_cluster(
          center=cluster_centers[2],
          scale=scale,
          num_points=samples_per_cluster),
      generate_random_cluster(
          center=cluster_centers[3],
          scale=scale,
          num_points=samples_per_cluster),
  ))

  # Cluster labels: 0, 1, 2 and 3
  labels = np.concatenate((
      np.zeros(samples_per_cluster),
      np.ones(samples_per_cluster),
      np.ones(samples_per_cluster) * 2,
      np.ones(samples_per_cluster) * 3,
  )).astype("uint8")

  return (features, labels)


# Hint: Play with "noise_scale" for different levels of overlap between
# the generated clusters. More noise makes the classification harder.
noise_scale = 2
training_features, training_labels = generate_features_and_labels(
    samples_per_cluster=250, scale=noise_scale)
test_features, test_labels = generate_features_and_labels(
    samples_per_cluster=250, scale=noise_scale)

num_clusters = int(round(np.max(training_labels))) + 1

# Hint: play with the number of layers to achieve different level of
# over-fitting and observe its effects on membership inference performance.
three_layer_model = keras.models.Sequential([
    layers.Dense(300, activation="relu"),
    layers.Dense(300, activation="relu"),
    layers.Dense(300, activation="relu"),
    layers.Dense(num_clusters, activation="relu"),
    layers.Softmax()
])
three_layer_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

two_layer_model = keras.models.Sequential([
    layers.Dense(300, activation="relu"),
    layers.Dense(300, activation="relu"),
    layers.Dense(num_clusters, activation="relu"),
    layers.Softmax()
])
two_layer_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


def crossentropy(true_labels, predictions):
  return keras.backend.eval(
      keras.losses.binary_crossentropy(
          keras.backend.variable(to_categorical(true_labels, num_clusters)),
          keras.backend.variable(predictions)))


def main(unused_argv):
  epoch_results = AttackResultsCollection([])

  num_epochs = 2
  models = {
      "two layer model": two_layer_model,
      "three layer model": three_layer_model,
  }
  for model_name in models:
    # Incrementally train the model and store privacy metrics every num_epochs.
    for i in range(1, 6):
      models[model_name].fit(
          training_features,
          to_categorical(training_labels, num_clusters),
          validation_data=(test_features,
                           to_categorical(test_labels, num_clusters)),
          batch_size=64,
          epochs=num_epochs,
          shuffle=True)

      training_pred = models[model_name].predict(training_features)
      test_pred = models[model_name].predict(test_features)

      # Add metadata to generate a privacy report.
      privacy_report_metadata = PrivacyReportMetadata(
          accuracy_train=metrics.accuracy_score(
              training_labels, np.argmax(training_pred, axis=1)),
          accuracy_test=metrics.accuracy_score(test_labels,
                                               np.argmax(test_pred, axis=1)),
          epoch_num=num_epochs * i,
          model_variant_label=model_name)

      attack_results = mia.run_attacks(
          AttackInputData(
              labels_train=training_labels,
              labels_test=test_labels,
              probs_train=training_pred,
              probs_test=test_pred,
              loss_train=crossentropy(training_labels, training_pred),
              loss_test=crossentropy(test_labels, test_pred)),
          SlicingSpec(entire_dataset=True, by_class=True),
          attack_types=(AttackType.THRESHOLD_ATTACK,
                        AttackType.LOGISTIC_REGRESSION),
          privacy_report_metadata=privacy_report_metadata)
      epoch_results.append(attack_results)

  # Generate privacy reports
  epoch_figure = privacy_report.plot_by_epochs(
      epoch_results, [PrivacyMetric.ATTACKER_ADVANTAGE, PrivacyMetric.AUC])
  epoch_figure.show()
  privacy_utility_figure = privacy_report.plot_privacy_vs_accuracy(
      epoch_results, [PrivacyMetric.ATTACKER_ADVANTAGE, PrivacyMetric.AUC])
  privacy_utility_figure.show()

  # Example of saving the results to the file and loading them back.
  with tempfile.TemporaryDirectory() as tmpdirname:
    filepath = os.path.join(tmpdirname, "results.pickle")
    attack_results.save(filepath)
    loaded_results = AttackResults.load(filepath)
    print(loaded_results.summary(by_slices=False))

  # Print attack metrics
  for attack_result in attack_results.single_attack_results:
    print("Slice: %s" % attack_result.slice_spec)
    print("Attack type: %s" % attack_result.attack_type)
    print("AUC: %.2f" % attack_result.roc_curve.get_auc())

    print("Attacker advantage: %.2f\n" %
          attack_result.roc_curve.get_attacker_advantage())

  max_auc_attacker = attack_results.get_result_with_max_auc()
  print("Attack type with max AUC: %s, AUC of %.2f" %
        (max_auc_attacker.attack_type, max_auc_attacker.roc_curve.get_auc()))

  max_advantage_attacker = attack_results.get_result_with_max_attacker_advantage(
  )
  print("Attack type with max advantage: %s, Attacker advantage of %.2f" %
        (max_advantage_attacker.attack_type,
         max_advantage_attacker.roc_curve.get_attacker_advantage()))

  # Print summary
  print("Summary without slices: \n")
  print(attack_results.summary(by_slices=False))

  print("Summary by slices: \n")
  print(attack_results.summary(by_slices=True))

  # Print pandas data frame
  print("Pandas frame: \n")
  pd.set_option("display.max_rows", None, "display.max_columns", None)
  print(attack_results.calculate_pd_dataframe())

  # Example of ROC curve plotting.
  figure = plotting.plot_roc_curve(
      attack_results.single_attack_results[0].roc_curve)
  figure.show()
  plt.show()

  # For saving a figure into a file:
  # plotting.save_plot(figure, <file_path>)

if __name__ == "__main__":
  app.run(main)
