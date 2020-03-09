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
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import split
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import PCA, StandardScaler

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    # Important: asNondeterministic requires Spark 2.3 or later
    # It can be safely removed i.e.
    # return udf(to_array_, ArrayType(DoubleType()))(col)
    # but at the cost of decreased performance
    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)

def main():
    spark = SparkSession.builder.appName('AmazonReviewsSparkProcessor').getOrCreate()
    
    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    
    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args['s3_input_data'].replace('s3://', 's3a://')
    s3_output_data = args['s3_output_data'].replace('s3://', 's3a://')

    schema = StructType([
        StructField('is_positive_sentiment', IntegerType(), True),
        StructField('review_body', StringType(), True)
    ])
    
    df_csv = spark.read.csv(path=s3_input_data,
                            schema=schema,
                            header=True,
                            quote=None)
    df_csv.show()
    
    tokenizer = Tokenizer(inputCol='review_body', outputCol='words')
    wordsData = tokenizer.transform(df_csv)
    
    hashingTF = HashingTF(inputCol='words', outputCol='raw_features', numFeatures=1000)
    featurizedData = hashingTF.transform(wordsData)
    
    # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    # 1) compute the IDF vector 
    # 2) scale the term frequencies by IDF
    # Therefore, we cache the result of the HashingTF transformation above to speed up the 2nd pass
    featurizedData.cache()

    # spark.mllib's IDF implementation provides an option for ignoring terms
    # which occur in less than a minimum number of documents.
    # In such cases, the IDF for these terms is set to 0.
    # This feature can be used by passing the minDocFreq value to the IDF constructor.
    idf = IDF(inputCol='raw_features', outputCol='features') #, minDocFreq=2)
    idfModel = idf.fit(featurizedData)
    features_df = idfModel.transform(featurizedData)
    features_df.select('is_positive_sentiment', 'features').show()

    num_features=300
    pca = PCA(k=num_features, inputCol='features', outputCol='pca_features')
    pca_model = pca.fit(features_df)
    pca_features_df = pca_model.transform(features_df).select('is_positive_sentiment', 'pca_features')
    pca_features_df.show(truncate=False)

    standard_scaler = StandardScaler(inputCol='pca_features', outputCol='scaled_pca_features')
    standard_scaler_model = standard_scaler.fit(pca_features_df)
    standard_scaler_features_df = standard_scaler_model.transform(pca_features_df).select('is_positive_sentiment', 'scaled_pca_features')
    standard_scaler_features_df.show(truncate=False)

    expanded_features_df = (standard_scaler_features_df.withColumn('f', to_array(col('scaled_pca_features')))
        .select(['is_positive_sentiment'] + [col('f')[i] for i in range(num_features)]))
    expanded_features_df.show()

    expanded_features_df.write.csv(path=s3_output_data,
                       header=None,
                       quote=None,
                       mode='overwrite')

    print('Wrote to train file:  {}'.format(s3_output_data))
        

if __name__ == "__main__":
    main()
