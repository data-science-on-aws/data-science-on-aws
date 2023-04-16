import functools
import multiprocessing

from datetime import datetime
import subprocess
import sys

## PIP INSTALLS ##
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1", "torchdata==0.5.1"])
import torch

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1", "datasets==2.9.0", "evaluate==0.4.0"])
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig
from datasets import Dataset

subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate==0.4.0", "py7zr==0.20.4", "sentencepiece", "rouge_score"])
import evaluate

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
    
    # extract the model tar file from the training job
    print("input_model: {}".format(args.input_model))
    print("Listing contents of input model dir: {}".format(args.input_model))
    input_files = os.listdir(args.input_model)
    for file in input_files:
        print(file)
    model_tar_path = "{}/model.tar.gz".format(args.input_model)
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(args.input_model)
    model_tar.close()

    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.input_model, use_fast=True)
    print(f'Loaded Local HuggingFace Tokenzier:\n{tokenizer}')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.input_model)
    print(f'Loaded Local HuggingFace Model:\n{model}')

    # List files in the input data
    print("input_data: {}".format(args.input_data))
    print("Listing contents of input data dir: {}".format(args.input_data))
    input_files = os.listdir(args.input_data)
    for file in input_files:
        print(file)

    # load the test dataset
    tokenized_dataset = load_dataset(
        args.input_data,
        data_files={'train': '*.parquet'}
    ).with_format("torch")
    print(f"Dataset loaded from local path {args.input_data}:\n{tokenized_dataset}")
    
    # load rouge metric
    rouge = evaluate.load('rouge')
    
    # select sample inputs for evaluation
    dialogues = tokenized_dataset['train'][0:10]['input_ids']
    baseline_summaries = tokenized_dataset['train'][0:10]['labels']

    # decode the original summaries
    human_baseline_summaries = []
    for base_summary in baseline_summaries:
        human_baseline_summaries.append(tokenizer.decode(base_summary, skip_special_tokens=True))

    # generate the tuned summaries
    tuned_outputs = model.generate(dialogues, GenerationConfig(max_new_tokens=200))
    tuned_model_summaries = []
    for tuned_summary in tuned_outputs:
        tuned_model_summaries.append(tokenizer.decode(tuned_summary, skip_special_tokens=True))
    
    # compute ROUGE metrics
    tuned_results = rouge.compute(
        predictions=tuned_model_summaries,
        references=human_baseline_summaries,
        use_aggregator=True,
        use_stemmer=True,
    )
    print(f'Fine-Tuned ROUGE metrics:\n{tuned_results}')
    
    # Model Output
    metrics_path = os.path.join(args.output_data, "metrics/")
    os.makedirs(metrics_path, exist_ok=True)

    report_dict = {
        "metrics": {
            "eval_rouge1": {
                "value": tuned_results['rouge1'],
            },
            "eval_rouge2": {
                "value": tuned_results['rouge2'],
            },
            "eval_rougeL": {
                "value": tuned_results['rougeL'],
            },
            "eval_rougeLsum": {
                "value": tuned_results['rougeLsum'],
            },
        },
    }
    print(f"Evalution Metric Report: \n{report_dict}")
    
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

    print("Evaluation Complete")


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
