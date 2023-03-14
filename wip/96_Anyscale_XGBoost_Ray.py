import ray
from ray.air.config import ScalingConfig
#from ray.data.preprocessors import MinMaxScaler
from ray.train.xgboost import XGBoostTrainer

# Initialize Ray runtime
ray.init()

########
# DATA #
########
# Read Parquet file to Ray Dataset
dataset = ray.data.read_parquet(
#    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxi_2021.parquet"
#    "s3://dsoaws/intro-to-ray-air/nyc_taxi_2021.parquet"
    's3://dsoaws/nyc-taxi-orig-cleaned-dropped-parquet-all-years-multiple-files-100GB/'
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
#    preprocessor=preprocessor,
)

# Invoke training - this is computationally intensive operation
# The resulting object grants access to metrics, checkpoints, and errors
result = trainer.fit()

# Report results
print(f"train rmse = {result.metrics['train-rmse']}")
#print(f"valid acc = {1 - result.metrics['valid-error']:.4f}")
print(f"iteration = {result.metrics['training_iteration']}")


