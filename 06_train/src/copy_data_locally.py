import boto3
import sagemaker
import pandas as pd

sess   = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

sm = boto3.Session().client(service_name='sagemaker', region_name=region)

prefix_train = 'feature-store/amazon-reviews/csv/balanced-tfidf-without-header/train'
prefix_validation = 'feature-store/amazon-reviews/csv/balanced-tfidf-without-header/validation'
prefix_test = 'feature-store/amazon-reviews/csv/balanced-tfidf-without-header/test'

balanced_tfidf_without_header_train_path = './{}'.format(prefix_train)
balanced_tfidf_without_header_validation_path = './{}'.format(prefix_validation)
balanced_tfidf_without_header_test_path = './{}'.format(prefix_test)

import os
os.makedirs(prefix_train, exist_ok=True)
os.makedirs(prefix_validation, exist_ok=True)
os.makedirs(prefix_test, exist_ok=True)

balanced_tfidf_without_header_train_s3_uri = 's3://{}/{}'.format(bucket, prefix_train)
balanced_tfidf_without_header_validation_s3_uri = 's3://{}/{}'.format(bucket, prefix_validation)
balanced_tfidf_without_header_test_s3_uri = 's3://{}/{}'.format(bucket, prefix_test)

from subprocess import call
call('aws s3 cp --recursive {} {}'.format(balanced_tfidf_without_header_train_s3_uri, balanced_tfidf_without_header_train_path), shell=True)
call('aws s3 cp --recursive {} {}'.format(balanced_tfidf_without_header_validation_s3_uri, balanced_tfidf_without_header_validation_path), shell=True)
call('aws s3 cp --recursive {} {}'.format(balanced_tfidf_without_header_test_s3_uri, balanced_tfidf_without_header_test_path), shell=True)
