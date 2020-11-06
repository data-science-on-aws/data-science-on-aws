import json
import datetime
import random
import boto3
import csv
import io

kinesis = boto3.client('kinesis', region_name='us-east-1') 

def lambda_handler(event, context):
    stream_name = "dsoaws-data-stream"
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket="sagemaker-us-east-1-806570384721", Key="data/amazon_reviews_us_Digital_Software_v1_00_noheader.csv")
    csv =  str(response['Body'].read().decode('UTF-8'))
    lines = csv.split("\n")
    for line in csv.split("\n"):
        val = line.split(",")
        data = json.dumps(getRating(val[0], val[1]))
        kinesis.put_record(StreamName=stream_name, Data=data, PartitionKey="reviews")
    return "complete"

def getRating(starRating, reviewBody):
    data = {}
    data['starRating'] = starRating
    data['reviewBody'] = reviewBody
    return data