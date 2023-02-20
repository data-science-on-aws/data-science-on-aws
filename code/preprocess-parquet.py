import argparse
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.7.2", "scikit-learn", "pyarrow", "pandas"])

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    VectorIndexer,
)
from pyspark.sql.functions import *

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()


    df_ride_fare = spark.read.option("recursiveFileLookup", "true").parquet(
        os.path.join("s3://", args.s3_input_bucket, args.s3_input_key_prefix, "ride-fare")
    )
    df_ride_fare.show()

        
    df_ride_info = spark.read.option("recursiveFileLookup", "true").parquet(
        os.path.join("s3://", args.s3_input_bucket, args.s3_input_key_prefix, "ride-info")
    )
    df_ride_info.show()

        
    df_joined = df_ride_fare.join(df_ride_info, on="ride_id")

    # (Optional) Calculate average total_amount per passenger count
    # df_avg_amount_by_passenger_count = df_joined.select("passenger_count", "total_amount") \
    #                                             .groupby("passenger_count") \
    #                                             .avg("total_amount") \
    #                                             .sort("passenger_count")
    # df_avg_amount_by_passenger_count.show()

    
    # Drop columns
    df_dropped = df_joined.drop("ride_id") \
                          .drop("pickup_at") \
                          .drop("dropoff_at") \
                          .drop("store_and_fwd_flag")
    df_dropped.show()

    
    # Split the dataset randomly into 70% for training and 30% for testing. Passing a seed for deterministic behavior
    df_train, df_validation = df_dropped.randomSplit([0.7, 0.3], seed = 42)
    print("There are %d train examples and %d validation examples." % (df_train.count(), df_validation.count()))

    # Write out the data
    df_train.write.mode("overwrite").parquet(os.path.join("s3://", args.s3_output_bucket, args.s3_output_key_prefix, "train"))
    df_validation.write.mode("overwrite").parquet(os.path.join("s3://", args.s3_output_bucket, args.s3_output_key_prefix, "validation"))

    
    ###############
    # This doesn't work because we can't `pip install xgboost` on all of the workers
    # using SageMaker PySparkProcessor without a bunch of shenanigans. Lots of open
    # github issues on this issue.
    ###############

#     # (Optional) Train the XGBoost model using XGBoost's 1.7+ support for Spark using Python 3.8+
#     from pyspark.ml.feature import VectorAssembler, VectorIndexer

#     # Remove the target column from the input feature set.
#     featuresCols = df_dropped.columns
#     featuresCols.remove("total_amount")

#     # vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
#     vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures", handleInvalid="skip")

#     # vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
#     vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=100, handleInvalid="skip")
#     #from sparkdl.xgboost import XgboostRegressor

#     # The next step is to define the model training stage of the pipeline. 
#     # The following command defines a XgboostRegressor model that learns to predict the labels in the `label_col` column.
#     #xgb_regressor = XgboostRegressor(num_workers=5, labelCol="total_amount", missing=0.0)


#     import xgboost
#     print(xgboost.__version__)

#     from xgboost.spark import SparkXGBRegressor

#     # The next step is to define the model training stage of the pipeline. 
#     # The following command defines a XgboostRegressor model that learns to predict the labels in the `label_col` column.
#     xgb_regressor = SparkXGBRegressor(label_col="total_amount", 
#                                       missing=0.0,
#                                       eta=0.2,
#                                       gamma=4,
#                                       max_depth=5,
#                                       min_child_weight=6,
#                                       num_round=50,
#                                       objective='reg:squarederror',
#                                       subsample=0.7)
    
#     from pyspark.ml.evaluation import RegressionEvaluator        
    
#     from pyspark.ml import Pipeline

#     pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, xgb_regressor])
    
#     pipelineModel = pipeline.fit(df_train)

#     predictions = pipelineModel.transform(df_validation)
    
#     evaluator = RegressionEvaluator(metricName="rmse",
#                                 labelCol=xgb_regressor.getLabelCol(),
#                                 predictionCol=xgb_regressor.getPredictionCol())
#     rmse = evaluator.evaluate(predictions)

#     print("RMSE on our validation set: %g" % rmse)
    
if __name__ == "__main__":
    main()