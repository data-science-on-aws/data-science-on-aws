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

    print('Listing contents of {}'.format(args.input_data))
#    dirs_input = os.listdir(args.input_data)

#    data = pd.concat([pd.read_csv(f, sep='\t', header=header) for f in glob.glob('{}/*.csv'.format(path))], ignore_index = True)

    # This would print all the files and directories
    for file in glob.glob('{}/*.tsv.gz'.format(args.input_data)):
        print(file)

        df = pd.read_csv(file, 
                         delimiter='\t', 
                         quoting=csv.QUOTE_NONE,
                         compression='gzip')
        df.shape

        df_scrubbed_raw = df

        df_scrubbed_raw['marketplace'] = df_scrubbed_raw['marketplace'].replace(',', ' ')
        df_scrubbed_raw['review_id'] = df_scrubbed_raw['review_id'].replace(',', ' ')
        df_scrubbed_raw['product_id'] = df_scrubbed_raw['product_id'].replace(',', ' ')
        df_scrubbed_raw['product_title'] = df_scrubbed_raw['product_title'].replace(',', ' ')
        df_scrubbed_raw['product_category'] = df_scrubbed_raw['product_category'].replace(',', ' ')
        df_scrubbed_raw['review_headline'] = df_scrubbed_raw['review_headline'].replace(',', ' ')
        df_scrubbed_raw['review_body'] = df_scrubbed_raw['review_body'].replace(',', ' ')
        df_scrubbed_raw['review_date'] = df_scrubbed_raw['review_date'].replace(',', ' ')

        df_scrubbed_raw.shape

        df_scrubbed_raw.head(5)

        df_scrubbed_raw.isna().values.any()

        df_scrubbed_raw = df_scrubbed_raw.dropna()
        df_scrubbed_raw = df_scrubbed_raw.reset_index(drop=True)
        df_scrubbed_raw.shape

        df_scrubbed_raw.head(5)

        df_is_positive_sentiment = (df_scrubbed_raw['star_rating'] >= 4).astype(int)
        df_scrubbed_raw.insert(0, 'is_positive_sentiment', df_is_positive_sentiment)
        df_scrubbed_raw.shape

        print('Creating directory {}'.format(args.output_data))
        os.makedirs(args.output_data, exist_ok=True)

        filename_without_extension = Path(Path(file).stem).stem

        print('Writing to {}/part-{}-{}.csv'.format(args.output_data, args.current_host, filename_without_extension))

        df_scrubbed_raw.to_csv('{}/part-{}-{}.csv'.format(args.output_data, args.current_host, filename_without_extension), sep=',', index=False, header=True)

#        with open('{}/part-{}-{}.csv'.format(args.output_data, args.current_host, file), 'w') as fd:
#            fd.write('host{},thanks,andre,and,alex!'.format(args.current_host))
#            fd.close()

    print('Listing contents of {}'.format(args.output_data))
    dirs_output = os.listdir(args.output_data)
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
