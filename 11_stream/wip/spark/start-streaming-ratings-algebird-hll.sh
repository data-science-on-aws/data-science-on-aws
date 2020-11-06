#!/bin/bash

mkdir -p ./logs

echo '...Starting Spark Streaming App:  Store Approx Rating Counts using Algebird HyperLogLog (distinct count)...'

#nohup 

./spark/spark-2.4.6-bin-without-hadoop/bin/spark-submit --packages org.slf4j:slf4j-simple:1.7.30 --jars spark-submit --jars ./spark/aws-java-sdk-dynamodb-1.11.722.jar,./spark/aws-java-sdk-core-1.11.722.jar,./spark/aws-java-sdk-s3-1.11.722.jar,./spark/aws-java-sdk-1.11.722.jar,./spark/hadoop-common-3.2.1.jar,./spark/hadoop-aws-3.2.1.jar,./spark/spark-2.4.6-bin-without-hadoop/jars/spark-repl_2.11-2.4.6.jar,./spark/spark-2.4.6-bin-without-hadoop/jars/spark-catalyst_2.11-2.4.6.jar,./spark/spark-2.4.6-bin-without-hadoop/jars/spark-sql_2.11-2.4.6.jar,./spark/spark-2.4.6-bin-without-hadoop/jars/spark-core_2.11-2.4.6.jar,./spark/spark-2.4.6-bin-without-hadoop/jars/spark-launcher_2.11-2.4.6.jar,./spark/spark-2.4.6-bin-without-hadoop/jars/spark-tags_2.11-2.4.6.jar,./jars/streaming_2.11-1.0.jar --class com.advancedspark.streaming.rating.approx.AlgebirdHyperLogLog 

# --repositories $SPARK_REPOSITORIES --jars $SPARK_SUBMIT_JARS --packages $SPARK_SUBMIT_PACKAGES 

# 2>&1 1>./logs/spark-streaming-ratings-algebird-hll.log 
#&

#echo '...logs available with tail -f ./logs/spark-streaming-ratings-algebird-hll.log | grep HLL'
