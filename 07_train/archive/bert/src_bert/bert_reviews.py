import os
import argparse
import csv
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import sklearn
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
import re
import glob
import json
import numpy as np
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==1.4.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torchvision'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'simpletransformers'])
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

import simpletransformers
from simpletransformers.classification import ClassificationModel

def load_dataset(path, sep, header):
    data = pd.concat([pd.read_csv(f, sep=sep, header=header) for f in glob.glob('{}/*.csv'.format(path))], ignore_index = True)

    labels = data.iloc[:,0]
    features = data.drop(data.columns[0], axis=1)

    if header==None:
        # Adjust the column names after dropped the 0th column above
        # New column names are 0 (inclusive) to len(features.columns) (exclusive)
        new_column_names = list(range(0, len(features.columns)))
        features.columns = new_column_names

    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='bert')
    parser.add_argument('--model-name', type=str, default='bert-base-cased')
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--train-data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args, _ = parser.parse_known_args()   
    model_type = args.model_type
    model_name = args.model_name
    backend = args.backend
    train_data = args.train_data
    validation_data = args.validation_data
    model_dir = args.model_dir
    hosts = args.hosts
    current_host = args.current_host
    num_gpus = args.num_gpus

    # TODO:  Convert to distributed data loader
    #        https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
    #        https://github.com/aws/sagemaker-python-sdk/issues/1110
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    print('Distributed training - {}'.format(is_distributed))
    use_cuda = args.num_gpus > 0
    print('Number of gpus available - {}'.format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        print('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # TODO:  Change this to use SM_CHANNEL_TRAIN and DistributedDataLoader, etc
    # X_train, y_train = load_dataset(train_data, ',', header=0)
    # X_validation, y_validation = load_dataset(validation_data, ',', header=0)

    df1 = pd.read_csv('./data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', 
                 delimiter='\t', 
                 quoting=csv.QUOTE_NONE,
                 compression='gzip',
                 header=0)[:100]
    print(df1.shape)

    df2 = pd.read_csv('./data/amazon_reviews_us_Video_Games_v1_00.tsv.gz',
                 delimiter='\t', 
                 quoting=csv.QUOTE_NONE,
                 compression='gzip', 
                 header=0)[:100]
    print(df2.shape)

    df = pd.concat([df1, df2])

    print('YES: {}'.format(df.isna().values.any()))
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Enrich the data
    df['is_positive_sentiment'] = (df['star_rating'] >= 4).astype(int)

    df_bert = df[['review_body', 'is_positive_sentiment']]
    df_bert.columns = ['text', 'labels']
    df_bert.head(5)

    print(df_bert.shape)

    df_bert = df_bert #[:200100]
    df_bert.shape

    from sklearn.model_selection import train_test_split

    df_bert_train, df_bert_holdout = train_test_split(df_bert, test_size=0.10)
    df_bert_validation, df_bert_test = train_test_split(df_bert_holdout, test_size=0.50)

    print(df_bert_train.shape)
    print(df_bert_validation.shape)
    print(df_bert_test.shape)

    # TODO:  change output_dir to SM_model_dir or output_path
    bert_args = {
       'output_dir': model_dir, 
       'cache_dir': 'cache/',
       'fp16': False,
       'max_seq_length': 128,
       'train_batch_size': 8,
       'eval_batch_size': 8,
       'gradient_accumulation_steps': 1,
       'num_train_epochs': 1,
       'weight_decay': 0,
       'learning_rate': 3e-5,
       'adam_epsilon': 1e-8,
       'warmup_ratio': 0.06,
       'warmup_steps': 0,
       'max_grad_norm': 1.0,
       'logging_steps': 50,
       'evaluate_during_training': False,
       'save_steps': 2000,
       'eval_all_checkpoints': True,
       'use_tensorboard': True,
       'tensorboard_dir': 'tensorboard',
       'overwrite_output_dir': True,
       'reprocess_input_data': False,
    }

    bert_model = ClassificationModel(model_type='distilbert', # bert, distilbert, etc, etc.
                                     model_name='distilbert-base-cased',
                                     args=bert_args,
                                     use_cuda=use_cuda)

    bert_model.train_model(train_df=df_bert_train,
                           eval_df=df_bert_validation,
                           show_running_loss=True)

    # TODO:  use the model_dir that is passed in through args
    #        (currently SM_MODEL_DIR)
#    os.makedirs(model_dir, exist_ok=True)
#    model_path = os.path.join(model_dir, 'bert-model')

#    pkl.dump(bert_model, open(model_path, 'wb'))
#    print('Wrote model to {}'.format(model_path))
   
#    result, model_outputs, wrong_predictions = bert_model.eval_model(eval_df=df_bert_test, acc=sklearn.metrics.accuracy_score)

#    print(result)

    # Show bad predictions
#    print('Number of wrong predictions: {}'.format(len(wrong_predictions)))
#    print('\n')

#    for prediction in wrong_predictions:
#        print(prediction.text_a)
#        print('\n')

#    predictions, raw_outputs = bert_model.predict(["""I really enjoyed this item.  I highly recommend it."""])

#    print('Predictions: {}'.format(predictions))
#    print('Raw outputs: {}'.format(raw_outputs))

#    predictions, raw_outputs = bert_model.predict(["""This item is awful and terrible."""])

#    print('Predictions: {}'.format(predictions))
#    print('Raw outputs: {}'.format(raw_outputs))
