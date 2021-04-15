# Tutorials

This folder contains a set of tutorials that demonstrate the features of this
library.
As demonstrated on MNIST in `mnist_dpsgd_tutorial.py`, the easiest way to use
a differentially private optimizer is to modify an existing TF training loop
to replace an existing vanilla optimizer with its differentially private
counterpart implemented in the library.

Here is a list of all the tutorials included:

* `lm_dpsgd_tutorial.py`: learn a language model with differential privacy.

* `mnist_dpsgd_tutorial.py`: learn a convolutional neural network on MNIST with
  differential privacy.

* `mnist_dpsgd_tutorial_eager.py`: learn a convolutional neural network on MNIST
  with differential privacy using Eager mode.

* `mnist_dpsgd_tutorial_keras.py`: learn a convolutional neural network on MNIST
  with differential privacy using tf.Keras.

* `mnist_lr_tutorial.py`: learn a differentially private logistic regression
  model on MNIST. The model illustrates application of the
  "amplification-by-iteration" analysis (https://arxiv.org/abs/1808.06651).

The rest of this README describes the different parameters used to configure
DP-SGD as well as expected outputs for the `mnist_dpsgd_tutorial.py` tutorial.

## Parameters

All of the optimizers share some privacy-specific parameters that need to
be tuned in addition to any existing hyperparameter. There are currently four:

* `learning_rate` (float): The learning rate of the SGD training algorithm. The
  higher the learning rate, the more each update matters. If the updates are noisy
  (such as when the additive noise is large compared to the clipping
  threshold), the learning rate must be kept low for the training procedure to converge.
* `num_microbatches` (int): The input data for each step (i.e., batch) of your
  original training algorithm is split into this many microbatches. Generally,
  increasing this will improve your utility but slow down your training in terms
  of wall-clock time. The total number of examples consumed in one global step
  remains the same. This number should evenly divide your input batch size.
* `l2_norm_clip` (float): The cumulative gradient across all network parameters
  from each microbatch will be clipped so that its L2 norm is at most this
  value. You should set this to something close to some percentile of what
  you expect the gradient from each microbatch to be. In previous experiments,
  we've found numbers from 0.5 to 1.0 to work reasonably well.
* `noise_multiplier` (float): This governs the amount of noise added during
  training. Generally, more noise results in better privacy and lower utility.
  This generally has to be at least 0.3 to obtain rigorous privacy guarantees,
  but smaller values may still be acceptable for practical purposes.

## Measuring Privacy

Differential privacy can be expressed using two values, epsilon and delta.
Roughly speaking, they mean the following:

* epsilon gives a ceiling on how much the probability of a particular output
  can increase by including (or removing) a single training example. We usually
  want it to be a small constant (less than 10, or, for more stringent privacy
  guarantees, less than 1). However, this is only an upper bound, and a large
  value of epsilon may still mean good practical privacy.
* delta bounds the probability of an arbitrary change in model behavior.
  We can usually set this to a very small number (1e-7 or so) without
  compromising utility. A rule of thumb is to set it to be less than the inverse
  of the training data size.

To find out the epsilon given a fixed delta value for your model, follow the
approach demonstrated in the `compute_epsilon` of the `mnist_dpsgd_tutorial.py`
where the arguments used to call the RDP accountant (i.e., the tool used to
compute the privacy guarantee) are:

* `q` : The sampling ratio, defined as (number of examples consumed in one
  step) / (total training examples).
* `noise_multiplier` : The noise_multiplier from your parameters above.
* `steps` : The number of global steps taken.

A detailed writeup of the theory behind the computation of epsilon and delta
is available at https://arxiv.org/abs/1908.10530.

## Expected Output

When the `mnist_dpsgd_tutorial.py` script is run with the default parameters,
the output will contain the following lines (leaving out a lot of diagnostic
info):
```
...
Test accuracy after 1 epochs is: 0.774
For delta=1e-5, the current epsilon is: 1.03
...
Test accuracy after 2 epochs is: 0.877
For delta=1e-5, the current epsilon is: 1.11
...
Test accuracy after 60 epochs is: 0.966
For delta=1e-5, the current epsilon is: 3.01
```

## Using Command-Line Interface for Privacy Budgeting

Before launching a (possibly quite lengthy) training procedure, it is possible
to compute, quickly and accurately, privacy loss at any point of the training.
To do so, run the script `privacy/analysis/compute_dp_sgd_privacy.py`, which
does not have any TensorFlow dependencies. For example, executing
```
compute_dp_sgd_privacy.py --N=60000 --batch_size=256 --noise_multiplier=1.1 --epochs=60 --delta=1e-5
```
allows us to conclude, in a matter of seconds, that DP-SGD run with default
parameters satisfies differential privacy with eps = 3.01 and delta = 1e-05.
Note that the flags provided in the command above correspond to the tutorial in
`mnist_dpsgd_tutorial.py`. The command is applicable to other datasets but the
values passed must be adapted (e.g., N the number of training points).


## Select Parameters

The table below has a few sample parameters illustrating various
accuracy/privacy tradeoffs achieved by the MNIST tutorial in
`mnist_dpsgd_tutorial.py` (default parameters are in __bold__; privacy epsilon
is reported at delta=1e-5; accuracy is averaged over 10 runs, its standard
deviation is less than .3% in all cases).

| Learning rate | Noise multiplier | Clipping threshold | Number of microbatches | Number of epochs | Privacy eps | Accuracy |
| ------------- | ---------------- | -----------------  | ---------------------- | ---------------- | ----------- | -------- |
| 0.1           |                  |                    | __256__                | 20               | no privacy  | 99.0%    |
| 0.25          | 1.3              | 1.5                | __256__                | 15               | 1.19        | 95.0%    |
| __0.15__      | __1.1__          | __1.0__            | __256__                |__60__            | 3.01        | 96.6%    |
| 0.25          | 0.7              | 1.5                | __256__                | 45               | 7.10        | 97.0%    |

