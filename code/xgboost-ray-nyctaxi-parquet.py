import os
import time
import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl
import glob

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

from sagemaker_ray_helper import RayHelper

import json
import pandas as pd
import xgboost as xgb

import ray
from ray import tune
from ray.air.checkpoint import Checkpoint
from ray.train.constants import TRAIN_DATASET_KEY
from ray.train.xgboost import XGBoostCheckpoint, XGBoostTrainer
from ray.air.config import ScalingConfig
from ray.data.preprocessor import Preprocessor


def train(args):
    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    
    print("Listing contents of {}".format(args.train))
    dirs_input = os.listdir(args.train)
    for file in dirs_input:
        print(file)

    import ray
    from ray.air.config import ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    ########
    # DATA #
    ########
    
    # Read Parquet file to Ray Dataset
    dataset = ray.data.read_parquet(args.train)

    # Split datasets into blocks for parallel preprocessing
    # `num_blocks` should be lower than number of cores in the cluster
    train_dataset = dataset.repartition(num_blocks=int(96/2))

    ############
    # TRAINING #
    ############

    # Create XGBoost trainer.
    # During training, it will use `num_blocks` workers.
    trainer = XGBoostTrainer(
        label_column="total_amount",
        num_boost_round=50,
        scaling_config=ScalingConfig(
            num_workers=49,
            resources_per_worker={'CPU': 90},            
            use_gpu=False,
        ),
        params={
            "eta": "0.2",
            "gamma": "4",
            "max_depth": "5",
            "min_child_weight": "6",
            "num_round": "50",
            "objective": "reg:squarederror",
            "subsample": "0.7",
            "verbosity": "2",
            "content_type":"parquet",
        },
        datasets={"train": train_dataset},
    )

    # Invoke training - this is computationally intensive operation
    # The resulting object grants access to metrics, checkpoints, and errors
    result = trainer.fit()
    
    # Report results
    print(f"train rmse = {result.metrics['train-rmse']}")
    #print(f"valid acc = {1 - result.metrics['valid-error']:.4f}")
    print(f"training_iteration = {result.metrics['training_iteration']}")

    print(result)
    
    # Need to write out the model    
    model_path = os.path.join(args.model_dir, "/xgb_model/")
    print(model_path)    
    
    result.checkpoint.to_directory(model_path)

    print("Listing contents of {}".format(model_path))
    dirs_input = os.listdir(model_path)
    for file in dirs_input:
        print(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--verbosity', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--tree_method', type=str, default="auto")
    parser.add_argument('--predictor', type=str, default="auto")
    parser.add_argument('--content_type', type=str, default="")

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    args, _ = parser.parse_known_args()

    ray_helper = RayHelper()
    
    ray_helper.start_ray()
    
    ray.cluster_resources()
    
    start = time.time()

    train(args)
    
    taken = time.time() - start
    
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")































