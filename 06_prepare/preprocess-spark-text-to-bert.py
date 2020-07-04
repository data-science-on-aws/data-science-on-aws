from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import collections
import subprocess
import sys
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pip', '--upgrade'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wrapt', '--upgrade', '--ignore-installed'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.1.0', '--ignore-installed'])
import tensorflow as tf
print(tf.__version__)
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==2.8.0'])
from transformers import DistilBertTokenizer

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import split
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# We set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
DATA_COLUMN = 'review_body'
LABEL_COLUMN = 'star_rating'
LABEL_VALUES = [1, 2, 3, 4, 5]

label_map = {}
for (i, label) in enumerate(LABEL_VALUES):
    label_map[label] = i


class InputFeatures(object):
  """BERT feature vectors."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class Input(object):
  """A single training/test input for sequence classification."""

  def __init__(self, text, label=None):
    """Constructs an Input.
    Args:
      text: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.text = text
    self.label = label
    

def convert_input(label, text):
    # First, we need to preprocess our data so that it matches the data BERT was trained on:
    #
    # 1. Lowercase our text (if we're using a BERT lowercase model)
    # 2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
    # 3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
    #
    # Fortunately, the Transformers tokenizer does this for us!
    #
#    tokens = tokenizer.tokenize(text_input.text)

    # Next, we need to do the following:
    #
    # 4. Map our words to indexes using a vocab file that BERT provides
    # 5. Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert))
    # 6. Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))
    #
    # Again, the Transformers tokenizer does this for us!
    #
    encode_plus_tokens = tokenizer.encode_plus(text,
                                               pad_to_max_length=True,
                                               max_length=MAX_SEQ_LENGTH)

    # Convert the text-based tokens to ids from the pre-trained BERT vocabulary
    input_ids = encode_plus_tokens['input_ids']
    # Specifies which tokens BERT should pay attention to (0 or 1)
    input_mask = encode_plus_tokens['attention_mask']
    # Segment Ids are always 0 for single-sequence tasks (or 1 if two-sequence tasks)
    segment_ids = [0] * MAX_SEQ_LENGTH

    # Label for our training data (star_rating 1 through 5)
    label_id = label_map[label]

    return {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'label_ids': [label_id]}


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(',')


def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open('/opt/ml/config/resourceconfig.json', 'r') as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print('/opt/ml/config/resourceconfig.json not found.  current_host is unknown.')
        pass # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--hosts', type=list_arg,
        default=resconfig.get('hosts', ['unknown']),
        help='Comma-separated list of host names running the job'
    )
    parser.add_argument('--current-host', type=str,
        default=resconfig.get('current_host', 'unknown'),
        help='Name of this host running the job'
    )
    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    return parser.parse_args()


def transform(spark, s3_input_data, s3_output_train_data, s3_output_validation_data, s3_output_test_data): 
    print('Processing {} => {}'.format(s3_input_data, s3_output_train_data, s3_output_validation_data, s3_output_test_data))
 
    schema = StructType([
        StructField('marketplace', StringType(), True),
        StructField('customer_id', StringType(), True),
        StructField('review_id', StringType(), True),
        StructField('product_id', StringType(), True),
        StructField('product_parent', StringType(), True),
        StructField('product_title', StringType(), True),
        StructField('product_category', StringType(), True),
        StructField('star_rating', IntegerType(), True),
        StructField('helpful_votes', IntegerType(), True),
        StructField('total_votes', IntegerType(), True),
        StructField('vine', StringType(), True),
        StructField('verified_purchase', StringType(), True),
        StructField('review_headline', StringType(), True),
        StructField('review_body', StringType(), True),
        StructField('review_date', StringType(), True)
    ])
    
    df_csv = spark.read.csv(path=s3_input_data,
                            sep='\t',
                            schema=schema,
                            header=True,
                            quote=None)
    df_csv.show()

    # This dataset should already be clean, but always good to double-check
    print('Showing null review_body rows...')
    df_csv.where(col('review_body').isNull()).show()

    print('Showing cleaned csv')
    df_csv_dropped = df_csv.na.drop(subset=['review_body'])
    df_csv_dropped.show()

    # TODO:  Balance
    
    features_df = df_csv_dropped.select(['star_rating', 'review_body'])
    features_df.show()

    tfrecord_schema = StructType([
      StructField("input_ids", ArrayType(IntegerType(), False)),
      StructField("input_mask", ArrayType(IntegerType(), False)),
      StructField("segment_ids", ArrayType(IntegerType(), False)),
      StructField("label_ids", ArrayType(IntegerType(), False))
    ])

    bert_transformer = udf(lambda text, label: convert_input(text, label), tfrecord_schema)

    spark.udf.register('bert_transformer', bert_transformer)

    transformed_df = features_df.select(bert_transformer('star_rating', 'review_body').alias('tfrecords'))
    transformed_df.show(truncate=False)

    flattened_df = transformed_df.select('tfrecords.*')
    flattened_df.show()

    # Split 90-5-5%
    train_df, validation_df, test_df = flattened_df.randomSplit([0.9, 0.05, 0.05])

    train_df.write.format('tfrecords').option('recordType', 'Example').save(path=s3_output_train_data)
    print('Wrote to output file:  {}'.format(s3_output_train_data))
    
    validation_df.write.format('tfrecords').option('recordType', 'Example').save(path=s3_output_validation_data)
    print('Wrote to output file:  {}'.format(s3_output_validation_data))

    test_df.write.format('tfrecords').option('recordType', 'Example').save(path=s3_output_test_data)    
    print('Wrote to output file:  {}'.format(s3_output_test_data))

    restored_test_df = spark.read.format('tfrecords').option('recordType', 'Example').load(path=s3_output_test_data)
    restored_test_df.show()


def main():
    spark = SparkSession.builder.appName('AmazonReviewsSparkProcessor').getOrCreate()

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args['s3_input_data'].replace('s3://', 's3a://')
    print(s3_input_data)

    s3_output_train_data = args['s3_output_train_data'].replace('s3://', 's3a://')
    print(s3_output_train_data)

    s3_output_validation_data = args['s3_output_validation_data'].replace('s3://', 's3a://')
    print(s3_output_validation_data)

    s3_output_test_data = args['s3_output_test_data'].replace('s3://', 's3a://')
    print(s3_output_test_data)

    transform(spark, s3_input_data, s3_output_train_data, s3_output_validation_data, s3_output_test_data)


if __name__ == "__main__":
    main()
