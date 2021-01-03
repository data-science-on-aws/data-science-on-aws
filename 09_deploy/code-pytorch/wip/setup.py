from setuptools import setup, find_packages

setup(name='sagemaker-bert-example',
      version='1.0',
      description='SageMaker Example for Bert.',
      author='sofian',
      author_email='hamitis@amazon.com',
      packages=find_packages(exclude=('tests', 'docs')))