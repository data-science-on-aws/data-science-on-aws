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

from xgboost_ray import train, RayDMatrix, RayParams #ray imports
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
    
    #input_files = glob.glob("{}/**/*.snappy.parquet".format(args.train))
    #input_files = glob.glob("{}/*/".format(args.train))
    #print('Input files: {}'.format(input_files))

    print("Listing contents of {}".format(args.train))
    dirs_input = os.listdir(args.train)
    for file in dirs_input:
        print(file)
    
    #dtrain = get_dmatrix(input_files, args.content_type)
    dtrain = get_dmatrix(args.train, args.content_type)
    if args.validation:
        dval = get_dmatrix(args.validation, args.content_type)
    else:
        dval = None    
    
    # watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective,
        'tree_method': args.tree_method,
        'predictor': args.predictor,
    }
    
    # train_path = os.environ.get("SM_CHANNEL_TRAIN") #, "airline_14col.data.bz2")
    # model_path = os.environ["SM_MODEL_DIR"]

    import ray
    from ray.air.config import ScalingConfig
    #from ray.data.preprocessors import MinMaxScaler
    from ray.train.xgboost import XGBoostTrainer

    # Initialize Ray runtime
#    ray.init()

    ########
    # DATA #
    ########
    
    # Read Parquet file to Ray Dataset
    dataset = ray.data.read_parquet(
        args.train
    #    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxi_2021.parquet"
    #    "s3://dsoaws/intro-to-ray-air/nyc_taxi_2021.parquet"
#        's3://dsoaws/nyc-taxi-orig-cleaned-dropped-parquet-all-years-multiple-files-100GB/'
    )

    # Split data into training and validation subsets
    #train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

    # Split datasets into blocks for parallel preprocessing
    # `num_blocks` should be lower than number of cores in the cluster
    train_dataset = dataset.repartition(num_blocks=12)
    #valid_dataset = valid_dataset.repartition(num_blocks=5)

    # Define a preprocessor to normalize the columns by their range
    #preprocessor = MinMaxScaler(columns=["trip_distance", "trip_duration"])

    ############
    # TRAINING #
    ############

    # Create XGBoost trainer.
    # During training, it will use `num_blocks` workers.
    trainer = XGBoostTrainer(
        label_column="total_amount",
        num_boost_round=50,
        scaling_config=ScalingConfig(
            use_gpu=False,  # True for the GPU training, 1 GPU per worker
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
        datasets={"train": train_dataset,
                  #"valid": valid_dataset
                 },
    )

    # Invoke training - this is computationally intensive operation
    # The resulting object grants access to metrics, checkpoints, and errors
    result = trainer.fit()
    
    # Report results
    print(f"train acc = {1 - result.metrics['train-error']:.4f}")
    #print(f"valid acc = {1 - result.metrics['valid-error']:.4f}")
    print(f"iteration = {result.metrics['training_iteration']}")

    print(result)
    # TODO:  Need to write out the model
    print(args.model_dir)



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

    start = time.time()
    train(args)
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")































