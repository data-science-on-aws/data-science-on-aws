from datetime import datetime
from sagemaker.tuner import ContinuousParameter

config = {}

config["job_level"] = {"region_name": "us-east-1", "run_hyperparameter_opt": "no"}

config["preprocess_data"] = {
    "s3_in_url": "s3://amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Software_v1_00.tsv.gz",
    "s3_out_bucket": "sagemaker-us-east-1-835319576252",  # replace
    "s3_out_prefix": "preprocess/",
    "delimiter": "\t",
}

config["prepare_data"] = {
    "s3_in_bucket": "sagemaker-us-east-1-835319576252",  # replace
    "s3_in_prefix": "preprocess/",
    "s3_out_bucket": "sagemaker-us-east-1-835319576252",  # replace
    "s3_out_prefix": "prepare/",
    "delimiter": "\t",
}

config["train_model"] = {
    "sagemaker_role": "TeamRole",
    "estimator_config": {
        "train_instance_count": 1,
        "train_instance_type": "ml.c5.4xlarge",
        "train_volume_size": 30,
        "train_max_run": 3600,
        "output_path": "s3://sagemaker-us-east-1-835319576252/train/",  # replace
        "base_job_name": "trng-recommender",
        "hyperparameters": {
            "feature_dim": "178729",
            "epochs": "10",
            "mini_batch_size": "200",
            "num_factors": "64",
            "predictor_type": "regressor",
        },
    },
    "inputs": {
        "train": "s3://sagemaker-us-east-1-835319576252/prepare/train/train.protobuf",  # replace
    },
}

config["tune_model"] = {
    "tuner_config": {
        "objective_metric_name": "test:rmse",
        "objective_type": "Minimize",
        "hyperparameter_ranges": {
            "factors_lr": ContinuousParameter(0.0001, 0.2),
            "factors_init_sigma": ContinuousParameter(0.0001, 1),
        },
        "max_jobs": 20,
        "max_parallel_jobs": 2,
        "base_tuning_job_name": "hpo-recommender",
    },
    "inputs": {
        "train": "s3://sagemaker-us-east-1-835319576252/prepare/train/train.protobuf",  # replace
        "test": "s3://sagemaker-us-east-1-835319576252/prepare/validate/validate.protobuf",  # replace
    },
}

config["batch_transform"] = {
    "transform_config": {
        "instance_count": 1,
        "instance_type": "ml.c4.xlarge",
        "data": "s3://sagemaker-us-east-1-835319576252/prepare/test/",
        "data_type": "S3Prefix",
        "content_type": "application/x-recordio-protobuf",
        "strategy": "MultiRecord",
        "output_path": "s3://sagemaker-us-east-1-835319576252/transform/",
    }
}
