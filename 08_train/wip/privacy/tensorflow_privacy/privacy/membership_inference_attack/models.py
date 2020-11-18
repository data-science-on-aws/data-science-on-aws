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
"""Trained models for membership inference attacks."""

from dataclasses import dataclass
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network

from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData


@dataclass
class AttackerData:
  """Input data for an ML classifier attack.

  This includes only the data, and not configuration.
  """

  features_train: np.ndarray = None
  # element-wise boolean array denoting if the example was part of training.
  is_training_labels_train: np.ndarray = None

  features_test: np.ndarray = None
  # element-wise boolean array denoting if the example was part of training.
  is_training_labels_test: np.ndarray = None


def create_attacker_data(attack_input_data: AttackInputData,
                         test_fraction: float = 0.25,
                         balance: bool = True) -> AttackerData:
  """Prepare AttackInputData to train ML attackers.

  Combines logits and losses and performs a random train-test split.

  Args:
    attack_input_data: Original AttackInputData
    test_fraction: Fraction of the dataset to include in the test split.
    balance: Whether the training and test sets for the membership inference
              attacker should have a balanced (roughly equal) number of samples
              from the training and test sets used to develop the model
              under attack.

  Returns:
    AttackerData.
  """
  attack_input_train = _column_stack(attack_input_data.logits_or_probs_train,
                                     attack_input_data.get_loss_train())
  attack_input_test = _column_stack(attack_input_data.logits_or_probs_test,
                                    attack_input_data.get_loss_test())

  if balance:
    min_size = min(attack_input_data.get_train_size(),
                   attack_input_data.get_test_size())
    attack_input_train = _sample_multidimensional_array(attack_input_train,
                                                        min_size)
    attack_input_test = _sample_multidimensional_array(attack_input_test,
                                                       min_size)

  features_all = np.concatenate((attack_input_train, attack_input_test))

  labels_all = np.concatenate(
      ((np.zeros(len(attack_input_train))), (np.ones(len(attack_input_test)))))

  # Perform a train-test split
  features_train, features_test, \
  is_training_labels_train, is_training_labels_test = \
    model_selection.train_test_split(
        features_all, labels_all, test_size=test_fraction, stratify=labels_all)
  return AttackerData(features_train, is_training_labels_train, features_test,
                      is_training_labels_test)


def _sample_multidimensional_array(array, size):
  indices = np.random.choice(len(array), size, replace=False)
  return array[indices]


def _column_stack(logits, loss):
  """Stacks logits and losses.

  In case that only one exists, returns that one.
  Args:
    logits: logits array
    loss: loss array

  Returns:
    stacked logits and losses (or only one if both do not exist).
  """
  if logits is None:
    return np.expand_dims(loss, axis=-1)
  if loss is None:
    return logits
  return np.column_stack((logits, loss))


class TrainedAttacker:
  """Base class for training attack models."""
  model = None

  def train_model(self, input_features, is_training_labels):
    """Train an attacker model.

    This is trained on examples from train and test datasets.
    Args:
      input_features : array-like of shape (n_samples, n_features) Training
        vector, where n_samples is the number of samples and n_features is the
        number of features.
      is_training_labels : a vector of booleans of shape (n_samples, )
        representing whether the sample is in the training set or not.
    """
    raise NotImplementedError()

  def predict(self, input_features):
    """Predicts whether input_features belongs to train or test.

    Args:
      input_features : A vector of features with the same semantics as x_train
        passed to train_model.
    Returns:
      An array of probabilities denoting whether the example belongs to test.
    """
    if self.model is None:
      raise AssertionError(
          'Model not trained yet. Please call train_model first.')
    return self.model.predict_proba(input_features)[:, 1]


class LogisticRegressionAttacker(TrainedAttacker):
  """Logistic regression attacker."""

  def train_model(self, input_features, is_training_labels):
    lr = linear_model.LogisticRegression(solver='lbfgs')
    param_grid = {
        'C': np.logspace(-4, 2, 10),
    }
    model = model_selection.GridSearchCV(
        lr, param_grid=param_grid, cv=3, n_jobs=1, verbose=0)
    model.fit(input_features, is_training_labels)
    self.model = model


class MultilayerPerceptronAttacker(TrainedAttacker):
  """Multilayer perceptron attacker."""

  def train_model(self, input_features, is_training_labels):
    mlp_model = neural_network.MLPClassifier()
    param_grid = {
        'hidden_layer_sizes': [(64,), (32, 32)],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    n_jobs = -1
    model = model_selection.GridSearchCV(
        mlp_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=0)
    model.fit(input_features, is_training_labels)
    self.model = model


class RandomForestAttacker(TrainedAttacker):
  """Random forest attacker."""

  def train_model(self, input_features, is_training_labels):
    """Setup a random forest pipeline with cross-validation."""
    rf_model = ensemble.RandomForestClassifier()

    param_grid = {
        'n_estimators': [100],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    n_jobs = -1
    model = model_selection.GridSearchCV(
        rf_model, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=0)
    model.fit(input_features, is_training_labels)
    self.model = model


class KNearestNeighborsAttacker(TrainedAttacker):
  """K nearest neighbor attacker."""

  def train_model(self, input_features, is_training_labels):
    knn_model = neighbors.KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7],
    }
    model = model_selection.GridSearchCV(
        knn_model, param_grid=param_grid, cv=3, n_jobs=1, verbose=0)
    model.fit(input_features, is_training_labels)
    self.model = model
