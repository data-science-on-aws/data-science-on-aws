import argparse
import json
import os
import pandas as pd
import csv
import glob
from pathlib import Path


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


def process(args):
    print('Current host: {}'.format(args.current_host))

#    print('Listing contents of {}'.format(args.input_data))
#    dirs_input = os.listdir(args.input_data)

    balanced_train_data = None
    balanced_validation_data = None
    balanced_test_data = None

    # This would print all the files and directories
    for file in glob.glob('{}/*.tsv.gz'.format(args.input_data)):
        print(file)

        df = pd.read_csv(file, 
                         delimiter='\t', 
                         quoting=csv.QUOTE_NONE,
                         compression='gzip')
        df.shape

        df_unbalanced_raw = df

        df_unbalanced_raw['marketplace'] = df_unbalanced_raw['marketplace'].replace(',', ' ')
        df_unbalanced_raw['review_id'] = df_unbalanced_raw['review_id'].replace(',', ' ')
        df_unbalanced_raw['product_id'] = df_unbalanced_raw['product_id'].replace(',', ' ')
        df_unbalanced_raw['product_title'] = df_unbalanced_raw['product_title'].replace(',', ' ')
        df_unbalanced_raw['product_category'] = df_unbalanced_raw['product_category'].replace(',', ' ')
        df_unbalanced_raw['review_headline'] = df_unbalanced_raw['review_headline'].replace(',', ' ')
        df_unbalanced_raw['review_body'] = df_unbalanced_raw['review_body'].replace(',', ' ')
        df_unbalanced_raw['review_date'] = df_unbalanced_raw['review_date'].replace(',', ' ')

        df_unbalanced_raw.shape

        df_unbalanced_raw.head(5)

        df_unbalanced_raw.isna().values.any()

        df_unbalanced_raw = df_unbalanced_raw.dropna()
        df_unbalanced_raw = df_unbalanced_raw.reset_index(drop=True)
        df_unbalanced_raw.shape

        df_unbalanced_raw.head(5)

        df_unbalanced_raw['is_positive_sentiment'] = (df_unbalanced_raw['star_rating'] >= 4).astype(int)            
#        df_is_positive_sentiment = (df_unbalanced_raw['star_rating'] >= 4).astype(int)
#        df_unbalanced_raw.insert(0, 'is_positive_sentiment', df_is_positive_sentiment)
        df_unbalanced_raw.shape

        # Split train, test, validation

        from sklearn.model_selection import train_test_split

        # Split all data into 90% train and 10% holdout
        df_unbalanced_raw_train, df_unbalanced_raw_holdout = train_test_split(df_unbalanced_raw, test_size=0.1, stratify=df_unbalanced_raw['is_positive_sentiment'])
        df_unbalanced_raw_train = df_unbalanced_raw_train.reset_index(drop=True)
        df_unbalanced_raw_holdout = df_unbalanced_raw_holdout.reset_index(drop=True)

        # Split the holdout into 50% validation and 50% test
        df_unbalanced_raw_validation, df_unbalanced_raw_test = train_test_split(df_unbalanced_raw_holdout, test_size=0.5, stratify=df_unbalanced_raw_holdout['is_positive_sentiment'])
        df_unbalanced_raw_validation = df_unbalanced_raw_validation.reset_index(drop=True)
        df_unbalanced_raw_test = df_unbalanced_raw_test.reset_index(drop=True)

        print('df_unbalanced_raw.shape={}'.format(df_unbalanced_raw.shape))
        print('df_unbalanced_raw_train.shape={}'.format(df_unbalanced_raw_train.shape))
        print('df_unbalanced_raw_validation.shape={}'.format(df_unbalanced_raw_validation.shape))
        print('df_unbalanced_raw_test.shape={}'.format(df_unbalanced_raw_test.shape))

        filename_without_extension = Path(Path(file).stem).stem

        # Balance the Dataset between Classes
        from sklearn.utils import resample

        is_negative_sentiment_df = df_unbalanced_raw.query('is_positive_sentiment == 0')
        is_positive_sentiment_df = df_unbalanced_raw.query('is_positive_sentiment == 1')

        # TODO:  check which sentiment has the least number of samples
        is_positive_downsampled_df = resample(is_positive_sentiment_df,
                                      replace = False,
                                      n_samples = len(is_negative_sentiment_df),
                                      random_state = 27)

        df_balanced_raw = pd.concat([is_negative_sentiment_df, is_positive_downsampled_df])
        df_balanced_raw = df_balanced_raw.reset_index(drop=True)

        df_balanced_raw.head(5)

        from sklearn.model_selection import train_test_split

        # Split all data into 90% train and 10% holdout
        df_balanced_raw_train, df_balanced_raw_holdout = train_test_split(df_balanced_raw, test_size=0.1, stratify=df_balanced_raw['is_positive_sentiment'])
        df_balanced_raw_train = df_balanced_raw_train.reset_index(drop=True)
        df_balanced_raw_holdout = df_balanced_raw_holdout.reset_index(drop=True)

        # Split the holdout into 50% validation and 50% test
        df_balanced_raw_validation, df_balanced_raw_test = train_test_split(df_balanced_raw_holdout, test_size=0.5, stratify=df_balanced_raw_holdout['is_positive_sentiment'])
        df_balanced_raw_validation = df_balanced_raw_validation.reset_index(drop=True)
        df_balanced_raw_test = df_balanced_raw_test.reset_index(drop=True)

        print('df_balanced_raw.shape={}'.format(df_balanced_raw.shape))
        print('df_balanced_raw_train.shape={}'.format(df_balanced_raw_train.shape))
        print('df_balanced_raw_validation.shape={}'.format(df_balanced_raw_validation.shape))
        print('df_balanced_raw_test.shape={}'.format(df_balanced_raw_test.shape))

        balanced_train_data = '{}/raw/labeled/split/balanced/header/train'.format(args.output_data)
        balanced_validation_data = '{}/raw/labeled/split/balanced/header/validation'.format(args.output_data)
        balanced_test_data = '{}/raw/labeled/split/balanced/header/test'.format(args.output_data)

        print('Creating directory {}'.format(balanced_train_data))
        os.makedirs(balanced_train_data, exist_ok=True)

        print('Creating directory {}'.format(balanced_validation_data))
        os.makedirs(balanced_validation_data, exist_ok=True)

        print('Creating directory {}'.format(balanced_test_data))
        os.makedirs(balanced_test_data, exist_ok=True)

        print('Writing to {}/part-{}-{}.csv'.format(balanced_train_data, args.current_host, filename_without_extension))
        df_balanced_raw_train.to_csv('{}/part-{}-{}.csv'.format(balanced_train_data, args.current_host, filename_without_extension), sep=',', index=False, header=True)

        print('Writing to {}/part-{}-{}.csv'.format(balanced_validation_data, args.current_host, filename_without_extension))      
        df_balanced_raw_validation.to_csv('{}/part-{}-{}.csv'.format(balanced_validation_data, args.current_host, filename_without_extension), sep=',', index=False, header=True)

        print('Writing to {}/part-{}-{}.csv'.format(balanced_test_data, args.current_host, filename_without_extension))
        df_balanced_raw_test.to_csv('{}/part-{}-{}.csv'.format(balanced_test_data, args.current_host, filename_without_extension), sep=',', index=False, header=True)

    print('Listing contents of {}'.format(args.output_data))
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(balanced_train_data))
    dirs_output = os.listdir(balanced_train_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(balanced_validation_data))
    dirs_output = os.listdir(balanced_validation_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(balanced_test_data))
    dirs_output = os.listdir(balanced_test_data)
    for file in dirs_output:
        print(file)

    print('Complete')
    
    
if __name__ == "__main__":
    args = parse_args()
    print('Loaded arguments:')
    print(args)
    
    print('Environment variables:')
    print(os.environ)

    process(args)
