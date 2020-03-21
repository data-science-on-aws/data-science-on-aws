from setuptools import setup, find_packages

setup(name='mnist_keras_tf2',
      version='1.0',
      description='SageMaker Example for MNIST Keras TensorFlow 2.x',
      author='cfregly',
      author_email='chris@fregly.com',
      url='https://github.com/data-science-on-aws',
      packages=find_packages(exclude=('tests', 'docs')))