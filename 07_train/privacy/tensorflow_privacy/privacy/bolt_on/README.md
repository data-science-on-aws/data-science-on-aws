# BoltOn Subpackage

This package contains source code for the BoltOn method, a particular
differential-privacy (DP) technique that uses output perturbations and
leverages additional assumptions to provide a new way of approaching the
privacy guarantees.

## BoltOn Description

This method uses 4 key steps to achieve privacy guarantees:
  1. Adds noise to weights after training (output perturbation).
  2. Projects weights to R, the radius of the hypothesis space,
      after each batch. This value is configurable by the user.
  3. Limits learning rate
  4. Uses a strongly convex loss function (see compile)

For more details on the strong convexity requirements, see:
Bolt-on Differential Privacy for Scalable Stochastic Gradient
Descent-based Analytics by Xi Wu et al. at https://arxiv.org/pdf/1606.04722.pdf

## Why BoltOn?

The major difference for the BoltOn method is that it injects noise post model
convergence, rather than noising gradients or weights during training. This
approach requires some additional constraints listed in the Description.
Should the use-case and model satisfy these constraints, this is another
approach that can be trained to maximize utility while maintaining the privacy.
The paper describes in detail the advantages and disadvantages of this approach
and its results compared to some other methods, namely noising at each iteration
and no noising.

## Tutorials

This package has a tutorial that can be found in the root tutorials directory,
under `bolton_tutorial.py`.

## Contribution

This package was initially contributed by Georgian Partners with the hope of
growing the tensorflow/privacy library. There are several rich use cases for
delta-epsilon privacy in machine learning, some of which can be explored here:
https://medium.com/apache-mxnet/epsilon-differential-privacy-for-machine-learning-using-mxnet-a4270fe3865e
https://arxiv.org/pdf/1811.04911.pdf

## Stability

As we are pegged on tensorflow2.0, this package may encounter stability
issues in the ongoing development of tensorflow2.0.

This sub-package is currently stable for 2.0.0a0, 2.0.0b0, and 2.0.0.b1 If you
would like to use this subpackage, please do use one of these versions as we
cannot guarantee it will work for all latest releases. If you do find issues,
feel free to raise an issue to the contributors listed below.

## Contacts

In addition to the maintainers of tensorflow/privacy listed in the root
README.md, please feel free to contact members of Georgian Partners. In
particular,

* Georgian Partners(@georgianpartners)
* Ji Chao Zhang(@Jichaogp)
* Christopher Choquette(@cchoquette)

## Copyright

Copyright 2019 - Google LLC
