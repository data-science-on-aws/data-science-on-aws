# TensorFlow Privacy

This repository contains the source code for TensorFlow Privacy, a Python
library that includes implementations of TensorFlow optimizers for training
machine learning models with differential privacy. The library comes with
tutorials and analysis tools for computing the privacy guarantees provided.

The TensorFlow Privacy library is under continual development, always welcoming
contributions. In particular, we always welcome help towards resolving the
issues currently open.

## Setting up TensorFlow Privacy

### Dependencies

This library uses [TensorFlow](https://www.tensorflow.org/) to define machine
learning models. Therefore, installing TensorFlow (>= 1.14) is a pre-requisite.
You can find instructions [here](https://www.tensorflow.org/install/). For
better performance, it is also recommended to install TensorFlow with GPU
support (detailed instructions on how to do this are available in the TensorFlow
installation documentation).

In addition to TensorFlow and its dependencies, other prerequisites are:

  * `scipy` >= 0.17

  * `mpmath` (for testing)

  * `tensorflow_datasets` (for the RNN tutorial `lm_dpsgd_tutorial.py` only)

### Installing TensorFlow Privacy

If you only want to use TensorFlow Privacy as a library, you can simply execute

`pip install tensorflow-privacy`

Otherwise, you can clone this GitHub repository into a directory of your choice:

```
git clone https://github.com/tensorflow/privacy
```

You can then install the local package in "editable" mode in order to add it to
your `PYTHONPATH`:

```
cd privacy
pip install -e .
```

If you'd like to make contributions, we recommend first forking the repository
and then cloning your fork rather than cloning this repository directly.

## Contributing

Contributions are welcomed! Bug fixes and new features can be initiated through
GitHub pull requests. To speed the code review process, we ask that:

*   When making code contributions to TensorFlow Privacy, you follow the `PEP8
    with two spaces` coding style (the same as the one used by TensorFlow) in
    your pull requests. In most cases this can be done by running `autopep8 -i
    --indent-size 2 <file>` on the files you have edited.

*   You should also check your code with pylint and TensorFlow's pylint
    [configuration file](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc)
    by running `pylint --rcfile=/path/to/the/tf/rcfile <edited file.py>`.

*   When making your first pull request, you
    [sign the Google CLA](https://cla.developers.google.com/clas)

*   We do not accept pull requests that add git submodules because of
    [the problems that arise when maintaining git submodules](https://medium.com/@porteneuve/mastering-git-submodules-34c65e940407)

## Tutorials directory

To help you get started with the functionalities provided by this library, we
provide a detailed walkthrough [here](tutorials/walkthrough/README.md) that
will teach you how to wrap existing optimizers
(e.g., SGD, Adam, ...) into their differentially private counterparts using
TensorFlow (TF) Privacy. You will also learn how to tune the parameters
introduced by differentially private optimization and how to
measure the privacy guarantees provided using analysis tools included in TF
Privacy.

In addition, the
`tutorials/` folder comes with scripts demonstrating how to use the library
features. The list of tutorials is described in the README included in the
tutorials directory.

NOTE: the tutorials are maintained carefully. However, they are not considered
part of the API and they can change at any time without warning. You should not
write 3rd party code that imports the tutorials and expect that the interface
will not break.

## Research directory

This folder contains code to reproduce results from research papers related to
privacy in machine learning. It is not maintained as carefully as the tutorials
directory, but rather intended as a convenient archive.

## TensorFlow 2.x

TensorFlow Privacy now works with TensorFlow 2! You can use the new
Keras-based estimators found in
`privacy/tensorflow_privacy/privacy/optimizers/dp_optimizer_keras.py`.

For this to work with `tf.keras.Model` and `tf.estimator.Estimator`, however,
you need to install TensorFlow 2.4 or later.

## Remarks

The content of this repository supersedes the following existing folder in the
tensorflow/models [repository](https://github.com/tensorflow/models/tree/master/research/differential_privacy)

## Contacts

If you have any questions that cannot be addressed by raising an issue, feel
free to contact:

* Galen Andrew (@galenmandrew)
* Steve Chien (@schien1729)
* Nicolas Papernot (@npapernot)

## Copyright

Copyright 2019 - Google LLC
