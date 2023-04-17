from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing
from datetime import datetime
from time import gmtime, strftime, sleep
import sys
import re
import collections
import argparse
import json
import os
import csv
import glob
from pathlib import Path
import time
import boto3
import subprocess
import pandas as pd
import re

## PIP INSTALLS ##
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1", "torchdata==0.5.1"])
import torch

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1", "datasets==2.9.0"])
import datasets
from transformers import AutoTokenizer

subprocess.check_call([sys.executable, "-m", "pip", "install", "promptsource==0.2.3"])
from promptsource.templates import DatasetTemplates


def _transform_to_dataset(file, output_data, balance_dataset, train_split_percentage, validation_split_percentage, test_split_percentage, model_checkpoint, dataset_templates_name, prompt_template_name):
    print("file {}".format(file))

    # Read the file
    #df = pd.read_parquet(file)
    df = pd.read_csv(file, delimiter="\t", quoting=csv.QUOTE_NONE, compression="gzip")

    df.isna().values.any()
    df = df.dropna()
    df = df.reset_index(drop=True)    

    print("Shape of dataframe {}".format(df.shape))

    # Balance
    if balance_dataset:
        # Balance the dataset down to the minority class
        df_grouped_by = df.groupby(["star_rating"]) 
        df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))

        df_balanced = df_balanced.reset_index(drop=True)
        print("Shape of balanced dataframe {}".format(df_balanced.shape))
        
        print(df_balanced["star_rating"].head(100))

        df = df_balanced

    # Split data    
    print("Shape of dataframe before splitting {}".format(df.shape))

    print("train split percentage {}".format(train_split_percentage))
    print("validation split percentage {}".format(validation_split_percentage))
    print("test split percentage {}".format(test_split_percentage))

    holdout_percentage = 1.00 - train_split_percentage
    print("validation holdout percentage {}".format(holdout_percentage))
    
    df_train, df_holdout = train_test_split(df, test_size=holdout_percentage)

    test_holdout_percentage = test_split_percentage / holdout_percentage
    
    print("test holdout percentage {}".format(test_holdout_percentage))
    
    df_validation, df_test = train_test_split(
        df_holdout, test_size=test_holdout_percentage)

    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("Shape of train dataframe {}".format(df_train.shape))
    print("Shape of validation dataframe {}".format(df_validation.shape))
    print("Shape of test dataframe {}".format(df_test.shape))

    # Convert Pandas dataframes into Datasets    
    from datasets import Dataset

    dataset_train = Dataset.from_pandas(df_train)
    dataset_validation = Dataset.from_pandas(df_validation)
    dataset_test = Dataset.from_pandas(df_test)    

    # Apply prompt  
    from promptsource.templates import DatasetTemplates
    prompt_templates = DatasetTemplates(dataset_templates_name)     
    prompt = prompt_templates[prompt_template_name]
    print(prompt.answer_choices)    
    print(prompt.__dict__)
    
    dataset_train = dataset_train.map(lambda row : {'prompt': 'PROMPT: ' + prompt.apply(row)[0] + '\nRESPONSE: ' + prompt.apply(row)[1] + '\n\n'})
    dataset_validation = dataset_validation.map(lambda row : {'prompt': 'PROMPT: ' + prompt.apply(row)[0] + '\nRESPONSE: ' + prompt.apply(row)[1] + '\n\n'})
    dataset_test = dataset_test.map(lambda row : {'prompt': 'PROMPT: ' + prompt.apply(row)[0] + '\nRESPONSE: ' + prompt.apply(row)[1] + '\n\n'})
    
    # Tokenize    
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    text_column_name = 'prompt'

    def tokenize_function(examples):
        tokenized = tokenizer(examples[text_column_name])
        return tokenized

    import multiprocessing

    num_cpus = multiprocessing.cpu_count()
    print('num_cpus {}'.format(num_cpus))

    # if using .tsv, the data will have `product_category`, but not `year`:  https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
    # if using .parquet, the data will have also have `year`:  https://s3.amazonaws.com/amazon-reviews-pds/readme.html
    tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True, num_proc=num_cpus, remove_columns=[
        'marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category',
        'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
        'review_headline', 'review_date', 'review_body', text_column_name]) # 'year'

    tokenized_dataset_validation = dataset_validation.map(tokenize_function, batched=True, num_proc=num_cpus, remove_columns=[
        'marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category',
        'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
        'review_headline', 'review_date', 'review_body', text_column_name]) # 'year'

    tokenized_dataset_test = dataset_validation.map(tokenize_function, batched=True, num_proc=num_cpus, remove_columns=[
        'marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category',
        'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
        'review_headline', 'review_date', 'review_body', text_column_name]) # 'year'

    
    # Group into blocks and save to S3/disk

    block_size = 128

    def group_texts(examples):    
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset_train = tokenized_dataset_train.map(
        group_texts,
        batched=True,
        batch_size=10,
        num_proc=num_cpus,
    )

    lm_dataset_validation = tokenized_dataset_validation.map(
       group_texts,
       batched=True,
       batch_size=10,
       num_proc=num_cpus,
    )
    
    lm_dataset_test = tokenized_dataset_test.map(
       group_texts,
       batched=True,
       batch_size=10,
       num_proc=num_cpus,
    )
    
    filename_without_extension = Path(Path(file).stem).stem
    
    os.makedirs('{}/train/'.format(output_data), exist_ok=True)
    os.makedirs('{}/validation/'.format(output_data), exist_ok=True)
    os.makedirs('{}/test/'.format(output_data), exist_ok=True)

    lm_dataset_train.to_parquet('{}/train/{}.parquet'.format(output_data, filename_without_extension))
    lm_dataset_validation.to_parquet('{}/validation/{}.parquet'.format(output_data, filename_without_extension))
    lm_dataset_validation.to_parquet('{}/test/{}.parquet'.format(output_data, filename_without_extension))


def process(args):

#    input_files = glob.glob("{}/*.parquet".format(args.input_data))
    input_files = glob.glob("{}/*.tsv.gz".format(args.input_data))
    print(input_files)

    print("Listing contents of {}".format(args.input_data))
    dirs_input = os.listdir(args.input_data)
    for file in dirs_input:
        print(file)

    train_data = "{}/train".format(args.output_data)
    validation_data = "{}/validation".format(args.output_data)
    test_data = "{}/test".format(args.output_data)

    print('train_data: {}'.format(train_data))
    print('validation_data: {}'.format(validation_data))
    print('test_data: {}'.format(test_data))
    
    transform_to_dataset = functools.partial(
        _transform_to_dataset,
        output_data=args.output_data,
        train_split_percentage=args.train_split_percentage, 
        validation_split_percentage=args.validation_split_percentage, 
        test_split_percentage=args.test_split_percentage,
        balance_dataset=args.balance_dataset,
        model_checkpoint=args.model_checkpoint,
        dataset_templates_name=args.dataset_templates_name,
        prompt_template_name=args.prompt_template_name,
        
    )

    num_cpus = multiprocessing.cpu_count()
    print("num_cpus {}".format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(transform_to_dataset, input_files)

    print("Listing contents of {}".format(args.output_data))
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(validation_data))
    dirs_output = os.listdir(validation_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(test_data))
    dirs_output = os.listdir(test_data)
    for file in dirs_output:
        print(file)
    

def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Process")

    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output/data",
    )
    parser.add_argument(
        "--train-split-percentage",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument("--balance-dataset", type=eval, default=True)
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="bigscience/bloom-560m"
    )
    parser.add_argument(
        "--dataset-templates-name",
        type=str,
        default="amazon_us_reviews/Wireless_v1_00",
    )
    parser.add_argument(
        "--prompt-template-name",
        type=str,
        default="Given the review body return a categorical rating",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
