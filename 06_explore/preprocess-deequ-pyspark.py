from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'pydeequ==0.1.5'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas==1.1.4'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'boto3==1.16.17'])

from io import StringIO
import boto3

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType

#def main():
from pyspark.sql.functions import *
from pydeequ.analyzers import *

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

#spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
#                                                  "org.apache.hadoop.mapred.FileOutputCommitter")

schema = StructType([
    StructField("marketplace", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("review_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("product_parent", StringType(), True),
    StructField("product_title", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("star_rating", IntegerType(), True),
    StructField("helpful_votes", IntegerType(), True),
    StructField("total_votes", IntegerType(), True),
    StructField("vine", StringType(), True),
    StructField("verified_purchase", StringType(), True),
    StructField("review_headline", StringType(), True),
    StructField("review_body", StringType(), True),
    StructField("review_date", StringType(), True)
])

dataset = spark.read.csv(s3_input_data, 
                         header=True,
                         schema=schema,
                         sep="\t",
                         quote="")

#     dataset = spark.read.option("sep", "\t")
#                             .option("header", "true")
#                             .option("quote", "")
#                             .schema(schema)
#                             .csv(s3_input_data)

# Calculate statistics on the dataset
analysisResult = AnalysisRunner(spark) \
                    .onData(dataset) \
                    .addAnalyzer(Size()) \
                    .addAnalyzer(Completeness("review_id")) \
                    .addAnalyzer(ApproxCountDistinct("review_id")) \
                    .addAnalyzer(Mean("star_rating")) \
                    .addAnalyzer(Compliance("top star_rating", "star_rating >= 4.0")) \
                    .addAnalyzer(Correlation("total_votes", "star_rating")) \
                    .addAnalyzer(Correlation("total_votes", "helpful_votes")) \
                    .run()

analysisResult_df = AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult)
analysisResult_df.show()

# Passing pandas=True in any call for getting metrics as DataFrames will return the dataframe in Pandas form! We'll see more of it down the line!
analysisResult_pd_df = AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult, pandas=True)
analysisResult_pd_df

# Check data quality
from pydeequ.checks import *
from pydeequ.verification import *

check = Check(spark, CheckLevel.Warning, "Amazon Customer Reviews")
checkResult = VerificationSuite(spark) \
    .onData(dataset) \
    .addCheck(
        check.hasSize(lambda x: x >= 200000) \
        .hasMin("star_rating", lambda x: x == 1.0) \
        .hasMax("star_rating", lambda x: x == 5.0)  \
        .isComplete("review_id")  \
        .isUnique("review_id")  \
        .isComplete("marketplace")  \
        .isContainedIn("marketplace", ["US", "UK", "DE", "JP", "FR"])) \
    .run()
            
print(f"Verification Run Status: {checkResult.status}")
checkResult_df = VerificationResult.checkResultsAsDataFrame(spark, checkResult)
checkResult_df.show(truncate=False)

checkResult_df.repartition(1).write.format('csv').option('header',True).mode('overwrite').option('sep','\t').save('{}/constraint-checks'.format(s3_output_analyze_data))

checkResult_df_pandas = VerificationResult.checkResultsAsDataFrame(spark, checkResult, pandas=True)
# csv_buffer = StringIO()
# checkResult_df_pandas.to_csv(csv_buffer)
# s3_resource = boto3.resource('s3')
# s3_resource.Object('sagemaker-us-east-1-835319576252', 'blahblah/output/constraint-checks').put(Body=csv_buffer.getvalue())

checkResult_success_df = VerificationResult.successMetricsAsDataFrame(spark, checkResult)
checkResult_success_df.show(truncate=False)

checkResult_success_df.repartition(1).write.format('csv').option('header',True).mode('overwrite').option('sep','\t').save('{}/success-metrics'.format(s3_output_analyze_data))

checkResult_success_df_pandas = VerificationResult.successMetricsAsDataFrame(spark, checkResult, pandas=True)
# csv_buffer = StringIO()
# checkResult_success_df_pandas.to_csv(csv_buffer)
# s3_resource = boto3.resource('s3')
# s3_resource.Object('sagemaker-us-east-1-835319576252', 'blahblah/output/success-metrics').put(Body=csv_buffer.getvalue())


# Suggest new checks and constraints
from pydeequ.suggestions import *

suggestionResult = ConstraintSuggestionRunner(spark) \
             .onData(dataset) \
             .addConstraintRule(DEFAULT()) \
             .run()

# Constraint Suggestions
print(type(suggestionResult))
print(suggestionResult)

# Constraint Suggestions in JSON format
print(json.dumps(suggestionResult, indent=2))    

# We can now investigate the constraints that Deequ suggested. 
#     suggestionsDataFrame = suggestionsResult['constraint_suggestions'].flatMap(lambda row: row { 
#           case (column, suggestions) => 
#             suggestions.map { constraint =>
#               (column, constraint.description, constraint.codeForConstraint)
#             } 
#     }.toSeq.toDS()


#if __name__ == "__main__":
    #main()
