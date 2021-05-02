from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

import pandas as pd
from datetime import datetime
import argparse
import subprocess
import sys

import os
import re
import collections
import json
import csv
import glob
from pathlib import Path

DATA_COLUMN = 'review_body'
LABEL_COLUMN = 'sentiment'
LABEL_VALUES = [-1, 0, 1]

# LABEL_MAP = {-1:0, 0:1, 1:2}
LABEL_MAP = {}
for (i, label) in enumerate(LABEL_VALUES):
    LABEL_MAP[label] = i

    
class Input(object):
  """A single training/test input for sequence classification."""

  def __init__(self, text, label=None):
    """Constructs an Input.
    Args:
      text: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and validation examples, but not for test examples.
    """
    self.text = text
    self.label = label
    
    
def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(',')


def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open('/opt/ml/config/resourceconfig.json', 'r') as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print('/opt/ml/config/resourceconfig.json not found. current_host is unknown.')
        pass # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--hosts', type=list_arg,
        default=resconfig.get('hosts', ['unknown']),
        help='Comma-separated list of host names running the job'
    )
    parser.add_argument('--current-host', type=str,
        default=resconfig.get('current_host', 'unknown'),
        help='Name of this host running the job'
    )
    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    parser.add_argument('--train-split-percentage', type=float,
        default=0.90,
    )
    parser.add_argument('--validation-split-percentage', type=float,
        default=0.05,
    )    
    parser.add_argument('--test-split-percentage', type=float,
        default=0.05,
    )
    parser.add_argument('--balance-dataset', type=eval,
        default=True
    )
    return parser.parse_args()



def to_sentiment(star_rating):
    star_rating = int(star_rating)
    # negative
    if star_rating <= 2:
        return -1
    # neutral
    elif star_rating == 3:
        return 0
    # positive
    else: 
        return 1

    
def _preprocess_file(file, balance_dataset):
    
    print('file {}'.format(file))
    print('balance_dataset {}'.format(balance_dataset))

    filename_without_extension = Path(Path(file).stem).stem

    ########### Read File
    df = pd.read_csv(file, 
                     delimiter='\t', 
                     quoting=csv.QUOTE_NONE,
                     compression='gzip')

    df.isna().values.any()
    df = df.dropna()
    df = df.reset_index(drop=True)
    print('Shape of dataframe {}'.format(df.shape))

    ########### Convert Star Rating Into Sentiment
    df['sentiment'] = df.star_rating.apply(lambda star_rating: to_sentiment(star_rating=star_rating))
    print('Shape of dataframe with sentiment {}'.format(df.shape))

    ########### Drop columns
    df = df[['sentiment','review_body']]
    df = df.reset_index(drop=True)

    print('Shape of dataframe after dropping columns {}'.format(df.shape))
    
    ########### Balance dataset
    if balance_dataset:  
        # Balance the dataset down to the minority class

        df_negative = df.query('sentiment == -1')
        df_neutral = df.query('sentiment == 0')
        df_positive = df.query('sentiment == 1')

        minority_count = min(df_negative.shape[0], 
                             df_neutral.shape[0], 
                             df_positive.shape[0]) 

        df_negative = resample(df_negative,
                                replace = False,
                                n_samples = minority_count,
                                random_state = 27)
        print('df_negative.shape: {}'.format(df_negative.shape))

        df_neutral = resample(df_neutral,
                                replace = False,
                                n_samples = minority_count,
                                random_state = 27)
        print('df_neutral.shape: {}'.format(df_neutral.shape))

        df_positive = resample(df_positive,
                                 replace = False,
                                 n_samples = minority_count,
                                 random_state = 27)
        print('df_positive.shape: {}'.format(df_positive.shape))

        df_balanced = pd.concat([df_negative, df_neutral, df_positive])
        df_balanced = df_balanced.reset_index(drop=True) 
        
        print('Shape of balanced df: {}'.format(df_balanced.shape))
        print(df_balanced['sentiment'].head())

        df = df_balanced
    
    ########### Split dataset
    print('Shape of dataframe before splitting {}'.format(df.shape))
    
    print('train split percentage {}'.format(args.train_split_percentage))
    print('validation split percentage {}'.format(args.validation_split_percentage))
    print('test split percentage {}'.format(args.test_split_percentage))    
    
    holdout_percentage = 1.00 - args.train_split_percentage
    print('holdout percentage {}'.format(holdout_percentage))
    df_train, df_holdout = train_test_split(df, 
                                            test_size=holdout_percentage, 
                                            stratify=df['sentiment'])

    test_holdout_percentage = args.test_split_percentage / holdout_percentage
    print('test holdout percentage {}'.format(test_holdout_percentage))
    df_validation, df_test = train_test_split(df_holdout, 
                                              test_size=test_holdout_percentage,
                                              stratify=df_holdout['sentiment'])
    
    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print('Shape of train dataframe {}'.format(df_train.shape))
    print('Shape of validation dataframe {}'.format(df_validation.shape))
    print('Shape of test dataframe {}'.format(df_test.shape))


    train_data = '{}/sentiment/train'.format(args.output_data)
    validation_data = '{}/sentiment/validation'.format(args.output_data)
    test_data = '{}/sentiment/test'.format(args.output_data)
    
    ########### Write TSV Files
    df_train.to_csv('{}/part-{}-{}.tsv'.format(train_data, args.current_host, filename_without_extension), sep='\t')
    df_validation.to_csv('{}/part-{}-{}.tsv'.format(validation_data, args.current_host, filename_without_extension), sep='\t')
    df_test.to_csv('{}/part-{}-{}.tsv'.format(test_data, args.current_host, filename_without_extension), sep='\t')


def process(args):
    print('Current host: {}'.format(args.current_host))
    
    preprocessed_data = '{}/sentiment'.format(args.output_data)
    train_data = '{}/sentiment/train'.format(args.output_data)
    validation_data = '{}/sentiment/validation'.format(args.output_data)
    test_data = '{}/sentiment/test'.format(args.output_data)
    
    preprocess_file = functools.partial(_preprocess_file,                 
                                        balance_dataset=args.balance_dataset)
    
    input_files = glob.glob('{}/*.tsv.gz'.format(args.input_data))

    num_cpus = multiprocessing.cpu_count()
    print('num_cpus {}'.format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(preprocess_file, input_files)

    print('Listing contents of {}'.format(preprocessed_data))
    dirs_output = os.listdir(preprocessed_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(validation_data))
    dirs_output = os.listdir(validation_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(test_data))
    dirs_output = os.listdir(test_data)
    for file in dirs_output:
        print(file)

    print('Complete')
    

if __name__ == "__main__":
    
    args = parse_args()
    print('Loaded arguments:')
    print(args)
    
    print('Environment variables:')
    print(os.environ)

    process(args)