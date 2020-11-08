import argparse
import bz2
import json
import os
import pickle
import random
import tempfile
import urllib.request
import pandas as pd
import glob
import pickle as pkl

import xgboost

from smdebug import SaveConfig
from smdebug.xgboost import Hook

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.05)  # 0.2
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--silent", type=int, default=0)
    parser.add_argument("--objective", type=str, default="multi:softmax")
    parser.add_argument("--num_class", type=int, default=15)
    parser.add_argument("--num_round", type=int, default=10)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    train_files_path, validation_files_path = args.train, args.validation
    
    train_files_list = glob.glob(train_files_path + '/*.*')
    print(train_files_list)
    
    val_files_list = glob.glob(validation_files_path + '/*.*')
    print(val_files_list)
    
    print('Loading training data...')
    df_train = pd.concat(map(pd.read_csv, train_files_list))
    print('Loading validation data...')
    df_val = pd.concat(map(pd.read_csv, val_files_list))
    print('Data loading completed.')
    
    y = df_train.Target.values
    X =  df_train.drop(['Target'], axis=1).values
    val_y = df_val.Target.values
    val_X = df_val.drop(['Target'], axis=1).values

    dtrain = xgboost.DMatrix(X, label=y)
    dval = xgboost.DMatrix(val_X, label=val_y)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "silent": args.silent,
        "objective": args.objective,
        "num_class": args.num_class}

    hook = Hook.create_from_json_file()
    hook.train_data = dtrain
    hook.validation_data = dval
    
    watchlist = [(dtrain, "train"), (dval, "validation")]

    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        callbacks=[hook])
    
    model_dir = os.environ.get('SM_MODEL_DIR')
    pkl.dump(bst, open(model_dir + '/model.bin', 'wb'))

if __name__ == "__main__":
    main()