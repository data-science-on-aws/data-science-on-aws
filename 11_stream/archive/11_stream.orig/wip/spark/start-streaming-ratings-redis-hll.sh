#!/bin/bash

echo '...Starting Spark Streaming App:  Store Ratings in Redis...'
  spark-submit \
    --packages org.apache.spark:spark-streaming-kafka-0-10_2.11:2.3.0 \
    --class com.advancedspark.streaming.rating.approx.RedisHyperLogLog \
    ./target/scala-2.11/streaming_2.11-1.0.jar
