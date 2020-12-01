import argparse
import json
import os
import pandas as pd
import numpy as np
import csv
import glob
from pathlib import Path

# requirements.txt
import subprocess
import sys
# this doesn't work
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'simpletransformers==0.22.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorboardx==2.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==1.4.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torchvision==0.5.0'])

import torch
import transformers as ppb
import sklearn
from sklearn.model_selection import train_test_split

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


def process(args):
    print('Current host: {}'.format(args.current_host))

#    print('Listing contents of {}'.format(args.input_data))
#    dirs_input = os.listdir(args.input_data)

    train_data = None
    validation_data = None
    test_data = None

    # This would print all the files and directories
    for file in glob.glob('{}/*.tsv.gz'.format(args.input_data)):
      print(file)

      filename_without_extension = Path(Path(file).stem).stem
      
      # chunksize=100 seems to work well 
      df_reader = pd.read_csv(file, 
                              delimiter='\t', 
                              quoting=csv.QUOTE_NONE,
                              compression='gzip',
                              chunksize=100)

      for df in df_reader:
        df.shape
        df.head(5)

        df.isna().values.any()

        df = df.dropna()
        df = df.reset_index(drop=True)
        df.shape

        df['is_positive_sentiment'] = (df['star_rating'] >= 4).astype(int)
        df.shape

        ###########
        # TODO:  increase batch size and run through all the data
        ###########        

        # Note:  we need to keep this at size 100
        batch_1 = df[['review_body', 'is_positive_sentiment']]
        batch_1.shape
        batch_1.head(5)

        # ## Loading the Pre-trained BERT model
        # Let's now load a pre-trained BERT model. 

        # For DistilBERT (lightweight for a notebook like this):
        #model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # For Bert (requires a lot more memory):
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        # Right now, the variable `model` holds a pretrained BERT or distilBERT model (a version of BERT that is smaller, but much faster and requiring a lot less memory.)
        # 
        # ## Preparing the Dataset
        # Before we can hand our sentences to BERT, we need to so some minimal processing to put them in the format it requires.
        # 
        # ### Tokenization
        # Our first step is to tokenize the sentences -- break them up into word and subwords in the format BERT is comfortable with.
        tokenized = batch_1['review_body'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        # ### Padding
        # After tokenization, `tokenized` is a list of sentences -- each sentences is represented as a list of tokens. We want BERT to process our examples all at once (as one batch). It's just faster that way. For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array, rather than a list of lists (of different lengths).

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

        # Our dataset is now in the `padded` variable, we can view its dimensions below:
        np.array(padded).shape

        # ### Masking
        # If we directly send `padded` to BERT, that would slightly confuse it. We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input. That's what attention_mask is:
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape

        # The `model()` function runs our sentences through BERT. The results of the processing will be returned into `last_hidden_states`.

        input_ids = torch.tensor(padded)  
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:,0,:].numpy()
        print(features)
        print(type(features))

        labels = batch_1['is_positive_sentiment']
        print(labels)
        print(type(labels))
        
        # TODO: Merge features and labels for our purpose here
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, stratify=batch_1['is_positive_sentiment'])

#        # Split all data into 90% train and 10% holdout
#        train_features, holdout_features, train_labels, holdout_labels = train_test_split(features, labels, stratify=batch_1['is_positive_sentiment'])
#        # Split the holdout into 50% validation and 50% test
#        validation_features, test_features, validation_labels, test_labels = train_test_split(holdout_features, holdout_labels, stratify=batch_1['is_positive_sentiment'])
      
        train_features.shape
        print(train_features)
        print(type(train_features))
        df_train_features = pd.DataFrame(train_features)
        df_train_labels = pd.DataFrame(train_labels)
        df_train = pd.concat([df_train_features, df_train_labels], axis=1)
        print(df_train)

#        validation_features.shape
#        print(validation_features)
#        print(type(validation_features))
#        df_validation_features = pd.DataFrame(validation_features)

        test_features.shape
        print(test_features)
        print(type(test_features))
        df_test_features = pd.DataFrame(test_features)

        train_output_data = '{}/raw/labeled/split/balanced/header/train'.format(args.output_data)
        print('Creating directory {}'.format(train_output_data))
        os.makedirs(train_output_data, exist_ok=True)
        print('Writing to {}/part-{}-{}.csv'.format(train_output_data, args.current_host, filename_without_extension))
        df_train.to_csv('{}/part-{}-{}.csv'.format(train_output_data, args.current_host, filename_without_extension), sep=',', index=False, header=None)

#        validation_output_data = '{}/raw/labeled/split/balanced/header/validation'.format(args.output_data)
#        print('Creating directory {}'.format(validation_output_data))
#        os.makedirs(validation_output_data, exist_ok=True)
#        print('Writing to {}/part-{}-{}.csv'.format(validation_output_data, args.current_host, filename_without_extension))     
#        df_validation_features.to_csv('{}/part-{}-{}.csv'.format(validation_output_data, args.current_host, filename_without_extension), sep=',', index=False, header=None)

        test_output_data = '{}/raw/labeled/split/balanced/header/test'.format(args.output_data)
        print('Creating directory {}'.format(test_output_data))
        os.makedirs(test_output_data, exist_ok=True)
        print('Writing to {}/part-{}-{}.csv'.format(test_output_data, args.current_host, filename_without_extension))
        df_test_features.to_csv('{}/part-{}-{}.csv'.format(test_output_data, args.current_host, filename_without_extension), sep=',', index=False, header=None)
        

    print('Listing contents of {}'.format(args.output_data))
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

#    print('Listing contents of {}'.format(validation_data))
#    dirs_output = os.listdir(validation_data)
#    for file in dirs_output:
#        print(file)

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
