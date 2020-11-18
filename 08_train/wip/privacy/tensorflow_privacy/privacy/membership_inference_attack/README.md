# Membership inference attack

A good privacy-preserving model learns from the training data, but
doesn't memorize it. This library provides empirical tests for measuring
potential memorization.

Technically, the tests build classifiers that infer whether a particular sample
was present in the training set. The more accurate such classifier is, the more
memorization is present and thus the less privacy-preserving the model is.

The privacy vulnerability (or memorization potential) is measured
via the area under the ROC-curve (`auc`) or via max{|fpr - tpr|} (`advantage`)
of the attack classifier. These measures are very closely related.

The tests provided by the library are "black box". That is, only the outputs of
the model are used (e.g., losses, logits, predictions). Neither model internals
(weights) nor input samples are required.

## How to use

### Codelab

The easiest way to get started is to go through [the introductory codelab](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs/codelab.ipynb).
This trains a simple image classification model and tests it against a series
of membership inference attacks.

For a more detailed overview of the library, please check the sections below.

### Basic usage

The simplest possible usage is

```python
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData

# Suppose we have the labels as integers starting from 0
# labels_train  shape: (n_train, )
# labels_test  shape: (n_test, )

# Evaluate your model on training and test examples to get
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

attacks_result = mia.run_attacks(
    AttackInputData(
        loss_train = loss_train,
        loss_test = loss_test,
        labels_train = labels_train,
        labels_test = labels_test))
```

This example calls `run_attacks` with the default options to run a host of
(fairly simple) attacks behind the scenes (depending on which data is fed in),
and computes the most important measures.

> NOTE: The train and test sets are balanced internally, i.e., an equal number
> of in-training and out-of-training examples is chosen for the attacks
> (whichever has fewer examples). These are subsampled uniformly at random
> without replacement from the larger of the two.

Then, we can view the attack results by:

```python
print(attacks_result.summary())
# Example output:
# -> Best-performing attacks over all slices
#      THRESHOLD_ATTACK achieved an AUC of 0.60 on slice Entire dataset
#      THRESHOLD_ATTACK achieved an advantage of 0.22 on slice Entire dataset
```


### Advanced usage

Sometimes, we have more information about the data, such as the logits and the
labels,
and we may want to have finer-grained control of the attack, such as using more
complicated classifiers instead of the simple threshold attack, and looks at the
attack results by examples' class.
In thoses cases, we can provide more information to `run_attacks`.

```python
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
```

First, similar as before, we specify the input for the attack as an
`AttackInputData` object:

```python
# Evaluate your model on training and test examples to get
# logits_train  shape: (n_train, n_classes)
# logits_test  shape: (n_test, n_classes)
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

attack_input = AttackInputData(
    logits_train = logits_train,
    logits_test = logits_test,
    loss_train = loss_train,
    loss_test = loss_test,
    labels_train = labels_train,
    labels_test = labels_test)
```

Instead of `logits`, you can also specify
`probs_train` and `probs_test` as the predicted probabilty vectors of each
example.

Then, we specify some details of the attack.
The first part includes the specifications of the slicing of the data. For
example, we may want to evaluate the result on the whole dataset, or by class,
percentiles, or the correctness of the model's classification.
These can be specified by a `SlicingSpec` object.

```python
slicing_spec = SlicingSpec(
    entire_dataset = True,
    by_class = True,
    by_percentiles = False,
    by_classification_correctness = True)
```

The second part specifies the classifiers for the attacker to use.
Currently, our API supports five classifiers, including
`AttackType.THRESHOLD_ATTACK` for simple threshold attack,
`AttackType.LOGISTIC_REGRESSION`,
`AttackType.MULTI_LAYERED_PERCEPTRON`,
`AttackType.RANDOM_FOREST`, and
`AttackType.K_NEAREST_NEIGHBORS`
which use the corresponding machine learning models.
For some model, different classifiers can yield pertty different results.
We can put multiple classifers in a list:

```python
attack_types = [
    AttackType.THRESHOLD_ATTACK,
    AttackType.LOGISTIC_REGRESSION
]
```

Now, we can call the `run_attacks` methods with all specifications:

```python
attacks_result = mia.run_attacks(attack_input=attack_input,
                                 slicing_spec=slicing_spec,
                                 attack_types=attack_types)
```

This returns an object of type `AttackResults`. We can, for example, use the
following code to see the attack results specificed per-slice, as we have
request attacks by class and by model's classification correctness.

```python
print(attacks_result.summary(by_slices = True))
# Example output:
# ->  Best-performing attacks over all slices
#       THRESHOLD_ATTACK achieved an AUC of 0.75 on slice CORRECTLY_CLASSIFIED=False
#       THRESHOLD_ATTACK achieved an advantage of 0.38 on slice CORRECTLY_CLASSIFIED=False
#
#     Best-performing attacks over slice: "Entire dataset"
#       LOGISTIC_REGRESSION achieved an AUC of 0.61
#       THRESHOLD_ATTACK achieved an advantage of 0.22
#
#     Best-performing attacks over slice: "CLASS=0"
#       LOGISTIC_REGRESSION achieved an AUC of 0.62
#       LOGISTIC_REGRESSION achieved an advantage of 0.24
#
#     Best-performing attacks over slice: "CLASS=1"
#       LOGISTIC_REGRESSION achieved an AUC of 0.61
#       LOGISTIC_REGRESSION achieved an advantage of 0.19
#
#     ...
#
#     Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=True"
#       LOGISTIC_REGRESSION achieved an AUC of 0.53
#       THRESHOLD_ATTACK achieved an advantage of 0.05
#
#     Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=False"
#       THRESHOLD_ATTACK achieved an AUC of 0.75
#       THRESHOLD_ATTACK achieved an advantage of 0.38
```


### Viewing and plotting the attack results

We have seen an example of using `summary()` to view the attack results as text.
We also provide some other ways for inspecting the attack results.

To get the attack that achieves the maximum attacker advantage or AUC, we can do

```python
max_auc_attacker = attacks_result.get_result_with_max_auc()
max_advantage_attacker = attacks_result.get_result_with_max_attacker_advantage()
```
Then, for individual attack, such as `max_auc_attacker`, we can check its type,
attacker advantage and AUC by

```python
print("Attack type with max AUC: %s, AUC of %.2f, Attacker advantage of %.2f" %
      (max_auc_attacker.attack_type,
       max_auc_attacker.roc_curve.get_auc(),
       max_auc_attacker.roc_curve.get_attacker_advantage()))
# Example output:
# -> Attack type with max AUC: THRESHOLD_ATTACK, AUC of 0.75, Attacker advantage of 0.38
```
We can also plot its ROC curve by

```python
import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting

figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)
```

Additionally, we provide funcitonality to convert the attack results into Pandas
data frame:

```python
import pandas as pd

pd.set_option("display.max_rows", 8, "display.max_columns", None)
print(attacks_result.calculate_pd_dataframe())
# Example output:
#           slice feature slice value attack type  Attacker advantage       AUC
# 0        entire_dataset               threshold            0.216440  0.600630
# 1        entire_dataset                      lr            0.212073  0.612989
# 2                 class           0   threshold            0.226000  0.611669
# 3                 class           0          lr            0.239452  0.624076
# ..                  ...         ...         ...                 ...       ...
# 22  correctly_classfied        True   threshold            0.054907  0.471290
# 23  correctly_classfied        True          lr            0.046986  0.525194
# 24  correctly_classfied       False   threshold            0.379465  0.748138
# 25  correctly_classfied       False          lr            0.370713  0.737148
```


## Contact / Feedback

Fill out this
[Google form](https://docs.google.com/forms/d/1DPwr3_OfMcqAOA6sdelTVjIZhKxMZkXvs94z16UCDa4/edit)
or reach out to us at tf-privacy@google.com and let us know how you’re using
this module. We’re keen on hearing your stories, feedback, and suggestions!

## Contributing

If you wish to add novel attacks to the attack library, please check our
[guidelines](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/CONTRIBUTING.md).

## Copyright

Copyright 2020 - Google LLC
