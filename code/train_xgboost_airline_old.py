import os
import time

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

from xgboost_ray import train, RayDMatrix, RayParams #ray imports
from sagemaker_ray_helper import RayHelper
import ray


FILENAME = os.path.join(os.environ.get("SM_CHANNEL_TRAIN"), "airline_14col.data.bz2")
MODEL_DIR = os.environ["SM_MODEL_DIR"]

max_depth = 6
learning_rate = 0.1
min_split_loss = 0
min_weight = 1
l1_reg = 0
l2_reg = 1

def get_airline(num_rows=None):
    """
    Airline dataset (http://kt.ijs.si/elena_ikonomovska/data.html)
    Has categorical columns converted to ordinal and target variable "Arrival Delay" converted
    to binary target.
    - Dimensions: 115M rows, 13 columns.
    - Task: Binary classification
    :param num_rows:
    :return: X, y
    """

    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance":
            dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }

    df = pd.read_csv(FILENAME,
                     names=cols, dtype=dtype_columns, nrows=num_rows)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]

    del df
    return X, y



def main():
    
    X, y = get_airline()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.2)

    dtrain = RayDMatrix(X_train, y_train)
    dtest = RayDMatrix(X_test, y_test)
    dval = RayDMatrix(X_val, y_val)
    
    print("data loaded")
    
    config = {
        "max_depth": max_depth,
        'learning_rate': learning_rate, 'min_split_loss': min_split_loss,
        'min_child_weight': min_weight, 'alpha': l1_reg, 'lambda': l2_reg,
#        "tree_method": "gpu_hist",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    evals_result = {}

    start = time.time()
    bst = train(
        config,
        dtrain,
        evals_result=evals_result,
        # ray params custom for ray
        ray_params=RayParams(max_actor_restarts=1, 
                             num_actors=4,
                             #gpus_per_actor=1,
                             cpus_per_actor=32),
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")])
    
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    bst.save_model(f"{MODEL_DIR}/airline.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["val"]["error"][-1]))




if __name__ == "__main__":
    ray_helper = RayHelper()
    
    ray_helper.start_ray()

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")