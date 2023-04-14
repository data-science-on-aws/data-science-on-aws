import subprocess
import sys
import json
import argparse

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1", "datasets==2.9.0", "torch==1.13.1"])

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import os
import time


def transform_dataset(input_data,
                      output_data,
                      huggingface_model_name,
                      train_split_percentage,
                      test_split_percentage,
                      validation_split_percentage,
                      ):

    # load in the original dataset
    dataset = load_dataset(input_data)
    print(f'Dataset loaded from path: {input_data}\n{dataset}')
    
    # Load the tokenizer
    print(f'Loading the tokenizer for the model {huggingface_model_name}')
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
    
    # make train test validation split
    train_testvalid = dataset['train'].train_test_split(1 - train_split_percentage, seed=1234)
    test_valid = train_testvalid['test'].train_test_split(test_split_percentage / (validation_split_percentage + test_split_percentage), seed=1234)
    train_test_valid_dataset = DatasetDict(
        {
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']
        }
    )
    print(f'Dataset after splitting:\n{train_test_valid_dataset}')
    
    # create a tokenize function
    def tokenize_function(example):
        prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        inp = [prompt + i + end_prompt for i in example["dialogue"]]
        example['input_ids'] = tokenizer(inp, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
        return example
    
    # tokenize the dataset
    print(f'Tokenizing the dataset...')
    tokenized_datasets = train_test_valid_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
    print(f'Tokenizing complete!')
    
    # create directory for drop
    os.makedirs(f'{output_data}/train/', exist_ok=True)
    os.makedirs(f'{output_data}/test/', exist_ok=True)
    os.makedirs(f'{output_data}/validation/', exist_ok=True)
    file_root = str(int(time.time()*1000))
    
    # save the dataset to disk
    print(f'Writing the dataset to {output_data}')
    tokenized_datasets['train'].to_parquet(f'./{output_data}/train/{file_root}.parquet')
    tokenized_datasets['test'].to_parquet(f'./{output_data}/test/{file_root}.parquet')
    tokenized_datasets['validation'].to_parquet(f'./{output_data}/validation/{file_root}.parquet')
    print('Preprocessing complete!')

    
def process(args):

    print(f"Listing contents of {args.input_data}")
    dirs_input = os.listdir(args.input_data)
    for file in dirs_input:
        print(file)

    transform_dataset(input_data=args.input_data, #'./data-summarization/',
                      output_data=args.output_data, #'./data-summarization-processed/',
                      huggingface_model_name=args.model_checkpoint, #model_checkpoint,
                      train_split_percentage=args.train_split_percentage, #0.90
                      test_split_percentage=args.test_split_percentage, #0.05
                      validation_split_percentage=args.validation_split_percentage, #0.05
                     )

    print(f"Listing contents of {args.output_data}")
    dirs_output = os.listdir(args.output_data)
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
        default=0.85,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="google/flan-t5-base"
    )
    # parser.add_argument(
    #     "--dataset-templates-name",
    #     type=str,
    #     default="amazon_us_reviews/Wireless_v1_00",
    # )
    # parser.add_argument(
    #     "--prompt-template-name",
    #     type=str,
    #     default="Given the review body return a categorical rating",
    # )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
