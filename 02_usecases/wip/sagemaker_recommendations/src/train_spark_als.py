from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'


def main():
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    
    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args['s3_input_data'].replace('s3://', 's3a://')
    print(s3_input_data)
    s3_output_data = args['s3_output_data'].replace('s3://', 's3a://')
    print(s3_output_data)
    
    spark = SparkSession.builder \
        .appName("Spark_ALS") \
        .getOrCreate()

    lines = spark.read.text(s3_input_data).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), 
                                         movieId=int(p[1]),
                                         rating=float(p[2]), 
                                         timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=5, 
              regParam=0.01, 
              userCol="userId", 
              itemCol="movieId", 
              ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", 
                                    labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("\nrmse: " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    userRecs.show(truncate=True)

    # |userId|recommendations |
    # +------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    # |21 |[[29, 4.6440034], [2, 4.11408], [85, 3.541986], [74, 3.5180748], [53, 3.4512622], [62, 3.4409468], [94, 3.2867239], [31, 3.2672868], [76, 3.2666306], [58, 3.1715796]] |

    # Write top 10 movie recommendations for each user
    # Note:  This is commented out until we fix this: 
    #    org.apache.spark.sql.AnalysisException: CSV data source does not support array<struct<movieId:int,rating:float>> data type.;
    userRecs \
      .repartition(1) \
      .write \
      .mode("overwrite") \
      .json(f"{s3_output_data}/all-recommendations")
        
    # Generate top 10 movie recommendations for a specified set of 3 users
    # TODO:  Just select user_id "42"    
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    userSubsetRecs.show(truncate=False)
    
     # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)
    movieRecs.show()
    # Write top 10 user recommendations for each movie
    movieRecs \
      .repartition(1) \
      .write \
      .mode("overwrite") \
      .json(f"{s3_output_data}/top-10-recommendations")
  
#     # Generate top 10 user recommendations for a specified set of movies
#     # TODO:  Just select user_id "42"
#     movies = ratings.select(als.getItemCol()).distinct().limit(3)
#     movieSubSetRecs = model.recommendForItemSubset(movies, 10)
#     movieSubSetRecs.show(truncate=False)
        
    spark.stop()
    
    
if __name__ == "__main__":
    main()
