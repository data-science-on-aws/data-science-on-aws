# Amazon SageMaker Autopilot AutoML Notebook Helper Library

This package contains companion helper code that will help you abstract the complexity of low-level 
interaction with the Amazon SageMaker Python SDK for the Autopilot interactive workflow notebook. 
This is an open source library. Please feel free to modify and distribute the code.

## Workflow and features overview 

To help provide a smooth experience for the Autopilot interactive workflow notebook, this library 
contains the following high-level constructs:

`AutoMLInteractiveRunner` 


This is a convenient runner to help the user easily perform the following: Select AutoML candidates, 
execute AutoML feature engineering, prepare multiple-algorithm tuning parameters, and finally choose 
the right candidate for deployment. The runner keeps tracks of a dictionary of `AutoMLLocalCandidate` 
and manages operations where information from all candidates are needed. For example, 
multiple-algorithm tuning and model selections across all selected candidates.

`AutoMLLocalRunConfig`

This is a configuration class that keeps track of all input and output to Amazon Simple Storage Service 
(Amazon S3) paths, conventions, and AWS and Amazon SageMaker shared variables (e.g., session and roles) 
for an interactive execution of AutoMLInteractiveRunner.

`AutoMLLocalCandidate`

This class models a unique Autopilot candidate that is composed of a data transformer and an algorithm steps. 
The class stores all the required attributes to get the data transformer and algorithm retrained and 
it also provides help to get candidate steps and low-level SDK constructs for training. 
It keeps track of references to trained models and training jobs in memory.

`AutoMLCandidateAlgoStep` & `AutoMLCandidateDataTransformerStep`

These are stateless classes. They abstract the lowest level of details required to interact with the Amazon SageMaker 
framework containers for the data transformation and algorithm-tuning AutoML steps. 

## Requirements

The library is compatible with python 3.6+ and tested with IPython 6.4.0.
