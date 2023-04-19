from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "pydeequ==0.1.5"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==1.1.4"])

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType
from pyspark.sql.functions import *

from pydeequ.analyzers import *
from pydeequ.checks import *
from pydeequ.verification import *
from pydeequ.suggestions import *

# PySpark Deequ GitHub Repo:  https://github.com/awslabs/python-deequ


def main():
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args["s3_input_data"].replace("s3://", "s3a://")
    print(s3_input_data)
    s3_output_analyze_data = args["s3_output_analyze_data"].replace("s3://", "s3a://")
    print(s3_output_analyze_data)

    spark = SparkSession.builder.appName("PySparkAmazonReviewsAnalyzer").getOrCreate()

    schema = StructType(
        [
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
            StructField("review_date", StringType(), True),
        ]
    )

    dataset = spark.read.csv(s3_input_data, header=True, schema=schema, sep="\t", quote="")

    # Calculate statistics on the dataset
    analysisResult = (
        AnalysisRunner(spark)
        .onData(dataset)
        .addAnalyzer(Size())
        .addAnalyzer(Completeness("review_id"))
        .addAnalyzer(ApproxCountDistinct("review_id"))
        .addAnalyzer(Mean("star_rating"))
        .addAnalyzer(Compliance("top star_rating", "star_rating >= 4.0"))
        .addAnalyzer(Correlation("total_votes", "star_rating"))
        .addAnalyzer(Correlation("total_votes", "helpful_votes"))
        .run()
    )

    metrics = AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult)
    metrics.show(truncate=False)
    metrics.repartition(1).write.format("csv").mode("overwrite").option("header", True).option("sep", "\t").save(
        "{}/dataset-metrics".format(s3_output_analyze_data)
    )

    # Check data quality
    verificationResult = (
        VerificationSuite(spark)
        .onData(dataset)
        .addCheck(
            Check(spark, CheckLevel.Error, "Review Check")
            .hasSize(lambda x: x >= 200000)
            .hasMin("star_rating", lambda x: x == 1.0)
            .hasMax("star_rating", lambda x: x == 5.0)
            .isComplete("review_id")
            .isUnique("review_id")
            .isComplete("marketplace")
            .isContainedIn("marketplace", ["US", "UK", "DE", "JP", "FR"])
        )
        .run()
    )

    print(f"Verification Run Status: {verificationResult.status}")
    resultsDataFrame = VerificationResult.checkResultsAsDataFrame(spark, verificationResult)
    resultsDataFrame.show(truncate=False)
    resultsDataFrame.repartition(1).write.format("csv").mode("overwrite").option("header", True).option(
        "sep", "\t"
    ).save("{}/constraint-checks".format(s3_output_analyze_data))

    verificationSuccessMetricsDataFrame = VerificationResult.successMetricsAsDataFrame(spark, verificationResult)
    verificationSuccessMetricsDataFrame.show(truncate=False)
    verificationSuccessMetricsDataFrame.repartition(1).write.format("csv").mode("overwrite").option(
        "header", True
    ).option("sep", "\t").save("{}/success-metrics".format(s3_output_analyze_data))

    # Suggest new checks and constraints
    suggestionsResult = ConstraintSuggestionRunner(spark).onData(dataset).addConstraintRule(DEFAULT()).run()

    suggestions = suggestionsResult["constraint_suggestions"]
    parallelizedSuggestions = spark.sparkContext.parallelize(suggestions)

    suggestionsResultsDataFrame = spark.createDataFrame(parallelizedSuggestions)
    suggestionsResultsDataFrame.show(truncate=False)
    suggestionsResultsDataFrame.repartition(1).write.format("csv").mode("overwrite").option("header", True).option(
        "sep", "\t"
    ).save("{}/constraint-suggestions".format(s3_output_analyze_data))


#    spark.stop()


if __name__ == "__main__":
    main()
