from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import pandas as pd
from datetime import datetime
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==1.15.2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-hub==0.7.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bert-tensorflow==1.0.1'])

import tensorflow as tf
import tensorflow_hub as hub

print(tf.__version__)

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re

import argparse
import json
import os
import pandas as pd
import csv
import glob
from pathlib import Path

# Based on this...
# https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L479
    
# TODO:  Pass this into the processor
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

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
        print('/opt/ml/config/resourceconfig.json not found.  current_host is unknown.')
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
    return parser.parse_args()


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    
def process(args):
    print('Current host: {}'.format(args.current_host))
    
    train_output_data = None
    validation_output_data = None
    test_output_data = None

    # This would print all the files and directories
    for file in glob.glob('{}/*.tsv.gz'.format(args.input_data)):
        print(file)
        
        filename_without_extension = Path(Path(file).stem).stem
        
        df = pd.read_csv(file, 
                         delimiter='\t', 
                         quoting=csv.QUOTE_NONE,
                         compression='gzip')

        df.isna().values.any()
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        df['is_positive_sentiment'] = (df['star_rating'] >= 4).astype(int) 
        df.shape
        df.head(5)  

        # Balance the Dataset between Classes
        is_negative_sentiment_df = df.query('is_positive_sentiment == 0')
        is_positive_sentiment_df = df.query('is_positive_sentiment == 1')

        # TODO:  check which sentiment has the least number of samples
        is_positive_downsampled_df = resample(is_positive_sentiment_df,
                                      replace = False,
                                      n_samples = len(is_negative_sentiment_df),
                                      random_state = 27)

        df_balanced = pd.concat([is_negative_sentiment_df, is_positive_downsampled_df])
        df_balanced = df_balanced.reset_index(drop=True)
        df_balanced.shape
        df_balanced.head(5)
       
        # Split all data into 90% train and 10% holdout
        df_train, df_holdout = train_test_split(df_balanced, test_size=0.9, stratify=df_balanced['is_positive_sentiment'])
        df_validation, df_test = train_test_split(df_holdout, test_size=0.1, stratify=df_holdout['is_positive_sentiment'])

        df_train = df_train.reset_index(drop=True)
        df_validation = df_validation.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        
        DATA_COLUMN = 'review_body'
        LABEL_COLUMN = 'is_positive_sentiment'
        LABEL_VALUES = [0, 1]

        # #Data Preprocessing
        # We'll need to transform our data into a format BERT understands. This involves two steps. First, we create  `InputExample`'s using the constructor provided in the BERT library.
        # 
        # - `text_a` is the text we want to classify, which in this case, is the `Request` field in our Dataframe. 
        # - `text_b` is used if we're training a model to understand the relationship between sentences (i.e. is `text_b` a translation of `text_a`? Is `text_b` an answer to the question asked by `text_a`?). This doesn't apply to our task since we are predicting sentiment, so we can leave `text_b` blank.
        # - `label` is the label for our example (0 or 1)

        # Use the InputExample class from BERT's run_classifier code to create examples from the data
        train_InputExamples = df_train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Unused in this example
                                                                           text_a = x[DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[LABEL_COLUMN]), axis = 1)

        validation_InputExamples = df_validation.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                           text_a = x[DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[LABEL_COLUMN]), axis = 1)

        test_InputExamples = df_test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                           text_a = x[DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[LABEL_COLUMN]), axis = 1)

        # Next, we need to preprocess our data so that it matches the data BERT was trained on. For this, we'll need to do a couple of things (but don't worry--this is also included in the Python library):
        # 
        # 
        # 1. Lowercase our text (if we're using a BERT lowercase model)
        # 2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
        # 3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
        # 4. Map our words to indexes using a vocab file that BERT provides
        # 5. Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert))
        # 6. Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))
        # 
        # Happily, we don't have to worry about most of these details.
        # 
        # To start, we'll need to load a vocabulary file and lowercasing information directly from the BERT tf hub module:

        # This is a path to an uncased (all lowercase) version of BERT
        BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

        tokenizer = create_tokenizer_from_hub_module()

        # This BERT model expects lowercase data (that's what stored in tokenization_info["do_lower_case"]).
        # We also loaded BERT's vocab file. We also created a tokenizer, which breaks words into word pieces:

        tokenizer.tokenize("This here's an example of using the BERT tokenizer")

        # Using our tokenizer, we'll call `run_classifier.file_based_convert_examples_to_features` on our InputExamples to convert them into features BERT understands, then write the features to a file.

        # We'll set sequences to be at most 128 tokens long.
        MAX_SEQ_LENGTH = 128

        train_data = '{}/bert/labeled/split/balanced/header/train'.format(args.output_data)
        validation_data = '{}/bert/labeled/split/balanced/header/validation'.format(args.output_data, args.current_host, filename_without_extension)
        test_data = '{}/bert/labeled/split/balanced/header/test'.format(args.output_data, args.current_host, filename_without_extension)

        # Convert our train and validation features to InputFeatures (.tfrecord protobuf) that works with BERT and TensorFlow.
        df_train_embeddings = bert.run_classifier.file_based_convert_examples_to_features(train_InputExamples, LABEL_VALUES, MAX_SEQ_LENGTH, tokenizer, '{}/part-{}-{}.tfrecord'.format(train_data, args.current_host, filename_without_extension))

        df_validation_embeddings = bert.run_classifier.file_based_convert_examples_to_features(validation_InputExamples, LABEL_VALUES, MAX_SEQ_LENGTH, tokenizer, '{}/part-{}-{}.tfrecord'.format(validation_data, args.current_host, filename_without_extension))

        df_test_embeddings = bert.run_classifier.file_based_convert_examples_to_features(test_InputExamples, LABEL_VALUES, MAX_SEQ_LENGTH, tokenizer, '{}/part-{}-{}.tfrecord'.format(test_data, args.current_host, filename_without_extension))
        
                                                                                         
    print('Listing contents of {}'.format(args.output_data))
    dirs_output = os.listdir(args.output_data)
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
