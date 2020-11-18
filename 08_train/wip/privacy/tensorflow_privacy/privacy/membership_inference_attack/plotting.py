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
"""Plotting functionality for membership inference attack analysis.

Functions to plot ROC curves and histograms as well as functionality to store
figures to colossus.
"""

from typing import Text, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def save_plot(figure: plt.Figure, path: Text, outformat='png'):
  """Store a figure to disk."""
  if path is not None:
    with open(path, 'wb') as f:
      figure.savefig(f, bbox_inches='tight', format=outformat)
    plt.close(figure)


def plot_curve_with_area(x: Iterable[float],
                         y: Iterable[float],
                         xlabel: Text = 'x',
                         ylabel: Text = 'y') -> plt.Figure:
  """Plot the curve defined by inputs and the area under the curve.

  All entries of x and y are required to lie between 0 and 1.
  For example, x could be recall and y precision, or x is fpr and y is tpr.

  Args:
    x: Values on x-axis (1d)
    y: Values on y-axis (must be same length as x)
    xlabel: Label for x axis
    ylabel: Label for y axis

  Returns:
    The matplotlib figure handle
  """
  fig = plt.figure()
  plt.plot([0, 1], [0, 1], 'k', lw=1.0)
  plt.plot(x, y, lw=2, label=f'AUC: {metrics.auc(x, y):.3f}')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  return fig


def plot_histograms(train: Iterable[float],
                    test: Iterable[float],
                    xlabel: Text = 'x',
                    thresh: float = None) -> plt.Figure:
  """Plot histograms of training versus test metrics."""
  xmin = min(np.min(train), np.min(test))
  xmax = max(np.max(train), np.max(test))
  bins = np.linspace(xmin, xmax, 100)
  fig = plt.figure()
  plt.hist(test, bins=bins, density=True, alpha=0.5, label='test', log='y')
  plt.hist(train, bins=bins, density=True, alpha=0.5, label='train', log='y')
  if thresh is not None:
    plt.axvline(thresh, c='r', label=f'threshold = {thresh:.3f}')
  plt.xlabel(xlabel)
  plt.ylabel('normalized counts (density)')
  plt.legend()
  return fig


def plot_roc_curve(roc_curve) -> plt.Figure:
  """Plot the ROC curve and the area under the curve."""
  return plot_curve_with_area(
      roc_curve.fpr, roc_curve.tpr, xlabel='FPR', ylabel='TPR')
