from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DateType
from pyspark.sql.functions import *

def main():
    spark = SparkSession.builder.appName("AmazonReviewsSparkProcessor").getOrCreate()
    
    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    
    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args['s3_input_data'].replace('s3://', 's3a://')
    s3_output_train_data = args['s3_output_train_data'].replace('s3://', 's3a://')
    s3_output_validation_data = args['s3_output_validation_data'].replace('s3://', 's3a://')
    s3_output_test_data = args['s3_output_test_data'].replace('s3://', 's3a://')

    df = spark.read.parquet(s3_input_data) 
    df.createOrReplaceTempView('amazon_reviews_parquet_temp')

    df_sentiment_counts = spark.sql(' \
        SELECT COUNT(CASE WHEN star_rating >= 4 THEN 1 END) as count_positive_reviews, \
            COUNT(CASE WHEN star_rating < 4 THEN 1 END) as count_negative_reviews \
        FROM amazon_reviews_parquet_temp')
    
    ###########
    # See https://github.com/awslabs/amazon-sagemaker-examples/issues/994 for issues related to using /opt/ml/processing/output/
    ###########

    df_sentiment_counts_as_dict = df_sentiment_counts.collect()[0].asDict()
    
    count_positive_reviews = float(df_sentiment_counts_as_dict['count_positive_reviews'])
    print('count_positive_reviews {}'.format(count_positive_reviews))

    count_negative_reviews = float(df_sentiment_counts_as_dict['count_negative_reviews'])
    print('count_negative_reviews {}'.format(count_negative_reviews))    
    
    reviews_sample_percent = count_positive_reviews / count_negative_reviews
    print(reviews_sample_percent)
          
    df_sentiment = spark.sql(' \
        SELECT customer_id, \
               review_id, \
               product_id, \
               product_title, \
               review_headline, \
               review_body, \
               review_date, \
               year, \
               star_rating, \
               CASE \
                   WHEN star_rating >= 4 THEN 1 \
                   ELSE 0 \
               END AS sentiment, \
               product_category \
        FROM amazon_reviews_parquet_temp TABLESAMPLE BERNOULLI({}) \
    '.format(reviews_sample_percent))

    # Split the dataset into training, validation, and test datasets (90%, 5%, 5%)
    (train_df, validation_df, test_df) = df_sentiment.randomSplit([0.9, 0.05, 0.05])
    
    train_df.write.parquet(path=s3_output_train_data,
                           mode='overwrite')
    print('Wrote to train file:  {}'.format(s3_output_train_data))
        
    validation_df.write.parquet(path=s3_output_validation_data, 
                                mode='overwrite')
    print('Wrote to validation file:  {}'.format(s3_output_validation_data))
    
    test_df.write.parquet(path=s3_output_test_data, 
                          mode='overwrite')
    print('Wrote to test file:  {}'.format(s3_output_test_data))


if __name__ == "__main__":
    main()