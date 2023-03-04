import functools
import multiprocessing

from datetime import datetime
import subprocess
import sys

## PIP INSTALLS ##
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1", "torchdata==0.5.1"])
import torch

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1", "datasets==2.9.0"])
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
import datasets
from datasets import Dataset

#subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
import pandas as pd
import os
import re
import collections
import argparse
import json
import os
import numpy as np
import csv
import glob
from pathlib import Path
import tarfile
import itertools


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
        "--input-model",
        type=str,
        default="/opt/ml/processing/input/model",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )

    return parser.parse_args()



def process(args):
    print("Current host: {}".format(args.current_host))
    
    print("input_model: {}".format(args.input_model))
    print("Listing contents of input model dir: {}".format(args.input_model))
    input_files = os.listdir(args.input_model)
    for file in input_files:
        print(file)
    model_tar_path = "{}/model.tar.gz".format(args.input_model)
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(args.input_model)
    model_tar.close()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.input_model)
    
    print(model)

    ###########################################################################################
    # TODO:  Replace this with glob for all files and remove test_data/ from the model.tar.gz #
    ###########################################################################################
    #    evaluation_data_path = '/opt/ml/processing/input/data/'

    print("input_data: {}".format(args.input_data))
    print("Listing contents of input data dir: {}".format(args.input_data))
    input_files = os.listdir(args.input_data)
    for file in input_files:
        print(file)

    test_data_path = "{}/amazon_reviews_us_Digital_Software_v1_00.parquet".format(args.input_data)    
    print("Using just {} to evaluate.".format(test_data_path))

    # select 10 samples
    lm_dataset_test = Dataset.from_parquet(test_data_path).select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    print(lm_dataset_test.shape)

    training_args = TrainingArguments(
        "finetuned-amazon-customer-reviews",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01, 
        eval_steps=15,
        no_cuda=True    
    )    

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=lm_dataset_test,
    )

    evaluation_results = trainer.evaluate()
    print(evaluation_results)    

    # Model Output
    metrics_path = os.path.join(args.output_data, "metrics/")
    os.makedirs(metrics_path, exist_ok=True)

    report_dict = {
        "metrics": {
            "eval_loss": {
                "value": evaluation_results['eval_loss'],
            },
        },
    }

    evaluation_path = "{}/evaluation.json".format(metrics_path)
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    print("Listing contents of output dir: {}".format(args.output_data))
    output_files = os.listdir(args.output_data)
    for file in output_files:
        print(file)

    print("Listing contents of output/metrics dir: {}".format(metrics_path))
    output_files = os.listdir("{}".format(metrics_path))
    for file in output_files:
        print(file)

    print("Complete")


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
