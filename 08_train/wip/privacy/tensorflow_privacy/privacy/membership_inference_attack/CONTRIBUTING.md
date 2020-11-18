# How to Contribute

We are happy to accept contributions to this project under the research folder.
The research folder is intended for the attacks that are not yet generic enough
to be included into the main library.

We are happy to accept contributions to the primary codebase, see below for more
details.

Please follow these guidelines when sending us a pull request.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted
one (even if it was for a different project), you probably don't need to do it
again.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Does my new feature belong here?

### Research folder

We use the following principles to guide what we add to our libraries. If your
contribution doesn't align with these principles, we're likely to decline.

* **Novelty:** The code should provide new attacks to the library. We will not
accept code that duplicates existing attacks.
* **Appropriate context and explanation:** The code should contain a README.md
file based on the provided template.This template should explain the code's functionality, and provide basic steps on how to use it.
* **Experiment-driven:** The code should contain an runnable example or a colab (e.g. on a toy model such as MNIST or CIFAR-10).
* **Quality requirements:** (1) The code should adhere to the
[Google Python style guide](https://google.github.io/styleguide/pyguide).
(2) The public API of the attack should have clear code documentation (expected inputs/outputs)
(3) The code should have reasonable unit test coverage (>60%);

### Primary codebase

The primary codebase should include attacks that are of general interest and
have a wide range of applications. For example, the standard membership
inference test is applicable to virtually any classification model.

The code contributed to the primary codebase should have a production style API
which is consistent with the API of other attacks. Most likely, Google and the
contributing team will need to meet and discuss the API before starting the
contribution.


If you're uncertain whether a planned contribution fits with these principles,
[open an issue](https://github.com/tensorflow/privacy/issues/new)
and describe what you want to add. We'll let you know whether it's something we
want to include and will help you figure out the best way to implement it.

