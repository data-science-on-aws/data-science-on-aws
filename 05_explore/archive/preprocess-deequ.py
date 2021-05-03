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

def main():
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    
    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args['s3_input_data'].replace('s3://', 's3a://')
    print(s3_input_data)
    s3_output_analyze_data = args['s3_output_analyze_data'].replace('s3://', 's3a://')
    print(s3_output_analyze_data)
    
    spark = SparkSession.builder \
        .appName("Amazon_Reviews_Spark_Analyzer") \
        .getOrCreate()

    # Invoke Main from preprocess-deequ.jar
    getattr(spark._jvm.SparkAmazonReviewsAnalyzer, "run")(s3_input_data, s3_output_analyze_data)

if __name__ == "__main__":
    main()
