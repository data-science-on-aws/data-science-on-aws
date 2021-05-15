from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

from datetime import datetime
from time import gmtime, strftime, sleep

import sys
import re
import collections
import argparse
import json
import os
import csv
import glob
from pathlib import Path
import time
import boto3
import subprocess

## PIP INSTALLS ##
# This is 2.3.0 (vs. 2.3.1 everywhere else) because we need to
# use anaconda and anaconda only supports 2.3.0 at this time
subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "anaconda", "tensorflow==2.3.0", "-y"])
import tensorflow as tf
from tensorflow import keras

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "transformers==3.5.1", "-y"])
from transformers import DistilBertTokenizer
from transformers import DistilBertConfig

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker==2.24.1"])
import pandas as pd
import re
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

region = os.environ["AWS_DEFAULT_REGION"]
print("Region: {}".format(region))

#############################
## We may need to get the Role and Bucket before setting sm, featurestore_runtime, etc.
## Role and Bucket are malformed if we do this later.
sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)

caller_identity = sts.get_caller_identity()
print("caller_identity: {}".format(caller_identity))

assumed_role_arn = caller_identity["Arn"]
print("(assumed_role) caller_identity_arn: {}".format(assumed_role_arn))

assumed_role_name = assumed_role_arn.split("/")[-2]

iam = boto3.Session(region_name=region).client(service_name="iam", region_name=region)
get_role_response = iam.get_role(RoleName=assumed_role_name)
print("get_role_response {}".format(get_role_response))
role = get_role_response["Role"]["Arn"]
print("role {}".format(role))

bucket = sagemaker.Session().default_bucket()
print("The DEFAULT BUCKET is {}".format(bucket))
#############################

sm = boto3.Session(region_name=region).client(service_name="sagemaker", region_name=region)

featurestore_runtime = boto3.Session(region_name=region).client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)

s3 = boto3.Session(region_name=region).client(service_name="s3", region_name=region)

sagemaker_session = sagemaker.Session(
    boto_session=boto3.Session(region_name=region),
    sagemaker_client=sm,
    sagemaker_featurestore_runtime_client=featurestore_runtime,
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

REVIEW_BODY_COLUMN = "review_body"
REVIEW_ID_COLUMN = "review_id"
# DATE_COLUMN = 'date'

LABEL_COLUMN = "star_rating"
LABEL_VALUES = [1, 2, 3, 4, 5]

label_map = {}
for (i, label) in enumerate(LABEL_VALUES):
    label_map[label] = i


def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame


def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print("Feature Group status: {}".format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Feature Group status: {}".format(status))
        if status != "Created":
            print("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print("No feature group created yet.")


def create_or_load_feature_group(prefix, feature_group_name):

    # Feature Definitions for our records
    feature_definitions = [
        FeatureDefinition(feature_name="input_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="input_mask", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="segment_ids", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="label_id", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="review_id", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="date", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="label", feature_type=FeatureTypeEnum.INTEGRAL),
        #        FeatureDefinition(feature_name='review_body', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="split_type", feature_type=FeatureTypeEnum.STRING),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name, feature_definitions=feature_definitions, sagemaker_session=sagemaker_session
    )

    print("Feature Group: {}".format(feature_group))

    try:
        print(
            "Waiting for existing Feature Group to become available if it is being created by another instance in our cluster..."
        )
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        print("Before CREATE FG wait exeption: {}".format(e))
    #        pass

    try:
        record_identifier_feature_name = "review_id"
        event_time_feature_name = "date"

        print("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=f"s3://{bucket}/{prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False,
        )
        print("Creating Feature Group. Completed.")

        print("Waiting for new Feature Group to become available...")
        wait_for_feature_group_creation_complete(feature_group)
        print("Feature Group available.")
        feature_group.describe()

    except Exception as e:
        print("Exception: {}".format(e))

    return feature_group


class InputFeatures(object):
    """BERT feature vectors."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, review_id, date, label):
        #               review_body):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.review_id = review_id
        self.date = date
        self.label = label


#        self.review_body = review_body


class Input(object):
    """A single training/test input for sequence classification."""

    def __init__(self, text, review_id, date, label=None):
        """Constructs an Input.
        Args:
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text = text
        self.review_id = review_id
        self.date = date
        self.label = label


def convert_input(the_input, max_seq_length):
    # First, we need to preprocess our data so that it matches the data BERT was trained on:
    #
    # 1. Lowercase our text (if we're using a BERT lowercase model)
    # 2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
    # 3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
    #
    # Fortunately, the Transformers tokenizer does this for us!
    #
    tokens = tokenizer.tokenize(the_input.text)

    # Next, we need to do the following:
    #
    # 4. Map our words to indexes using a vocab file that BERT provides
    # 5. Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert))
    # 6. Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))
    #
    # Again, the Transformers tokenizer does this for us!
    #
    encode_plus_tokens = tokenizer.encode_plus(
        the_input.text,
        pad_to_max_length=True,
        max_length=max_seq_length,
        #                                               truncation=True
    )

    # The id from the pre-trained BERT vocabulary that represents the token.  (Padding of 0 will be used if the # of tokens is less than `max_seq_length`)
    input_ids = encode_plus_tokens["input_ids"]

    # Specifies which tokens BERT should pay attention to (0 or 1).  Padded `input_ids` will have 0 in each of these vector elements.
    input_mask = encode_plus_tokens["attention_mask"]

    # Segment ids are always 0 for single-sequence tasks such as text classification.  1 is used for two-sequence tasks such as question/answer and next sentence prediction.
    segment_ids = [0] * max_seq_length

    # Label for each training row (`star_rating` 1 through 5)
    label_id = label_map[the_input.label]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        review_id=the_input.review_id,
        date=the_input.date,
        label=the_input.label,
    )
    #        review_body=the_input.text)

    #     print('**input_ids**\n{}\n'.format(features.input_ids))
    #     print('**input_mask**\n{}\n'.format(features.input_mask))
    #     print('**segment_ids**\n{}\n'.format(features.segment_ids))
    #     print('**label_id**\n{}\n'.format(features.label_id))
    #     print('**review_id**\n{}\n'.format(features.review_id))
    #     print('**date**\n{}\n'.format(features.date))
    #     print('**label**\n{}\n'.format(features.label))
    #    print('**review_body**\n{}\n'.format(features.review_body))

    return features


def transform_inputs_to_tfrecord(inputs, output_file, max_seq_length):
    """Convert a set of `Input`s to a TFRecord file."""

    records = []

    tf_record_writer = tf.io.TFRecordWriter(output_file)

    for (input_idx, the_input) in enumerate(inputs):
        if input_idx % 10000 == 0:
            print("Writing input {} of {}\n".format(input_idx, len(inputs)))

        features = convert_input(the_input, max_seq_length)

        all_features = collections.OrderedDict()
        all_features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_ids))
        all_features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_mask))
        all_features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.segment_ids))
        all_features["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features.label_id]))

        tf_record = tf.train.Example(features=tf.train.Features(feature=all_features))
        tf_record_writer.write(tf_record.SerializeToString())

        records.append(
            {  #'tf_record': tf_record.SerializeToString(),
                "input_ids": features.input_ids,
                "input_mask": features.input_mask,
                "segment_ids": features.segment_ids,
                "label_id": features.label_id,
                "review_id": the_input.review_id,
                "date": the_input.date,
                "label": features.label,
                #                        'review_body': features.review_body
            }
        )

        #####################################
        ####### TODO:  REMOVE THIS BREAK #######
        #####################################
        # break

    tf_record_writer.close()

    return records


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Process")

    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--train-split-percentage",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument("--balance-dataset", type=eval, default=True)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--feature-store-offline-prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default=None,
    )

    return parser.parse_args()


def _transform_tsv_to_tfrecord(file, max_seq_length, balance_dataset, prefix, feature_group_name):
    print("file {}".format(file))
    print("max_seq_length {}".format(max_seq_length))
    print("balance_dataset {}".format(balance_dataset))
    print("prefix {}".format(prefix))
    print("feature_group_name {}".format(feature_group_name))

    # need to re-load since we can't pass feature_group object in _partial functions for some reason
    feature_group = create_or_load_feature_group(prefix, feature_group_name)

    filename_without_extension = Path(Path(file).stem).stem

    df = pd.read_csv(file, delimiter="\t", quoting=csv.QUOTE_NONE, compression="gzip")

    df.isna().values.any()
    df = df.dropna()
    df = df.reset_index(drop=True)

    print("Shape of dataframe {}".format(df.shape))

    if balance_dataset:
        # Balance the dataset down to the minority class
        df_grouped_by = df.groupby(["star_rating"]) 
        df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))

        df_balanced = df_balanced.reset_index(drop=True)
        print("Shape of balanced dataframe {}".format(df_balanced.shape))
        
        print(df_balanced["star_rating"].head(100))

        df = df_balanced

    print("Shape of dataframe before splitting {}".format(df.shape))

    print("train split percentage {}".format(args.train_split_percentage))
    print("validation split percentage {}".format(args.validation_split_percentage))
    print("test split percentage {}".format(args.test_split_percentage))

    holdout_percentage = 1.00 - args.train_split_percentage
    print("holdout percentage {}".format(holdout_percentage))
    
    df_train, df_holdout = train_test_split(df, test_size=holdout_percentage, stratify=df["star_rating"])

    test_holdout_percentage = args.test_split_percentage / holdout_percentage
    
    print("test holdout percentage {}".format(test_holdout_percentage))
    
    df_validation, df_test = train_test_split(
        df_holdout, test_size=test_holdout_percentage, stratify=df_holdout["star_rating"])

    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("Shape of train dataframe {}".format(df_train.shape))
    print("Shape of validation dataframe {}".format(df_validation.shape))
    print("Shape of test dataframe {}".format(df_test.shape))

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(timestamp)

    train_inputs = df_train.apply(
        lambda x: Input(
            label=x[LABEL_COLUMN], text=x[REVIEW_BODY_COLUMN], review_id=x[REVIEW_ID_COLUMN], date=timestamp
        ),
        axis=1,
    )

    validation_inputs = df_validation.apply(
        lambda x: Input(
            label=x[LABEL_COLUMN], text=x[REVIEW_BODY_COLUMN], review_id=x[REVIEW_ID_COLUMN], date=timestamp
        ),
        axis=1,
    )

    test_inputs = df_test.apply(
        lambda x: Input(
            label=x[LABEL_COLUMN], text=x[REVIEW_BODY_COLUMN], review_id=x[REVIEW_ID_COLUMN], date=timestamp
        ),
        axis=1,
    )

    # Next, we need to preprocess our data so that it matches the data BERT was trained on. For this, we'll need to do a couple of things (but don't worry--this is also included in the Python library):
    #
    #
    # 1. Lowercase our text (if we're using a BERT lowercase model)
    # 2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
    # 3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
    # 4. Map our words to indexes using a vocab file that BERT provides
    # 5. Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert))
    # 6. Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))
    #
    # We don't have to worry about these details.  The Transformers tokenizer does this for us.
    #
    train_data = "{}/bert/train".format(args.output_data)
    validation_data = "{}/bert/validation".format(args.output_data)
    test_data = "{}/bert/test".format(args.output_data)

    # Convert our train and validation features to InputFeatures (.tfrecord protobuf) that works with BERT and TensorFlow.
    train_records = transform_inputs_to_tfrecord(
        train_inputs,
        "{}/part-{}-{}.tfrecord".format(train_data, args.current_host, filename_without_extension),
        max_seq_length,
    )

    validation_records = transform_inputs_to_tfrecord(
        validation_inputs,
        "{}/part-{}-{}.tfrecord".format(validation_data, args.current_host, filename_without_extension),
        max_seq_length,
    )

    test_records = transform_inputs_to_tfrecord(
        test_inputs,
        "{}/part-{}-{}.tfrecord".format(test_data, args.current_host, filename_without_extension),
        max_seq_length,
    )

    df_train_records = pd.DataFrame.from_dict(train_records)
    df_train_records["split_type"] = "train"
    df_train_records.head()

    df_validation_records = pd.DataFrame.from_dict(validation_records)
    df_validation_records["split_type"] = "validation"
    df_validation_records.head()

    df_test_records = pd.DataFrame.from_dict(test_records)
    df_test_records["split_type"] = "test"
    df_test_records.head()

    # Add record to feature store
    df_fs_train_records = cast_object_to_string(df_train_records)
    df_fs_validation_records = cast_object_to_string(df_validation_records)
    df_fs_test_records = cast_object_to_string(df_test_records)

    print("Ingesting Features...")
    feature_group.ingest(data_frame=df_fs_train_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_validation_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_test_records, max_workers=3, wait=True)
    
    offline_store_status = None
    while offline_store_status != 'Active':
        try:
            offline_store_status = feature_group.describe()['OfflineStoreStatus']['Status']
        except:
            pass
        print('Offline store status: {}'.format(offline_store_status))    
    print('...features ingested!')


def process(args):
    print("Current host: {}".format(args.current_host))

    feature_group = create_or_load_feature_group(
        prefix=args.feature_store_offline_prefix, feature_group_name=args.feature_group_name
    )

    feature_group.describe()

    print(feature_group.as_hive_ddl())

    train_data = "{}/bert/train".format(args.output_data)
    validation_data = "{}/bert/validation".format(args.output_data)
    test_data = "{}/bert/test".format(args.output_data)

    transform_tsv_to_tfrecord = functools.partial(
        _transform_tsv_to_tfrecord,
        max_seq_length=args.max_seq_length,
        balance_dataset=args.balance_dataset,
        prefix=args.feature_store_offline_prefix,
        feature_group_name=args.feature_group_name,
    )

    input_files = glob.glob("{}/*.tsv.gz".format(args.input_data))

    num_cpus = multiprocessing.cpu_count()
    print("num_cpus {}".format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(transform_tsv_to_tfrecord, input_files)

    print("Listing contents of {}".format(args.output_data))
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(validation_data))
    dirs_output = os.listdir(validation_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(test_data))
    dirs_output = os.listdir(test_data)
    for file in dirs_output:
        print(file)

    offline_store_contents = None
    while offline_store_contents is None:
        objects_in_bucket = s3.list_objects(Bucket=bucket, Prefix=args.feature_store_offline_prefix)
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            print("Waiting for data in offline store...\n")
            sleep(60)

    print("Data available.")

    print("Complete")


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
