# This code is auto-generated.

import argparse
import importlib
import logging
import os
import shutil
import warnings

from joblib import dump

from sagemaker_sklearn_extension.externals import AutoMLTransformer
from sagemaker_sklearn_extension.externals.read_data import read_csv_data

# suppressing deprecated features warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(X, y, header, feature_transformer, label_transformer):
    """Trains the data processing model.

    Splits training data to features and labels based on the header (list of column names and target name).
    Creates an AutoMLTransformer with feature_transformer and label_transformer.
    Trains the model and returns the trained model.

    Parameters
    ----------
    X : array-like
        2D numpy array containing the feature data

    y : array-like
        1D numpy array target column

    header: sagemaker_sklearn_extension.externals.Header
        Object of class Header, used to map the column names to the appropriate index

    feature_transformer : obj
        transformer applied to features

    label_transformer: obj
        transformer applied to label

    Returns
    -------
    aml_tx : AutoMLTransformer
        trained data processing model

    """
    # create AutoMLTransformer
    aml_tx = AutoMLTransformer(
        header=header,
        feature_transformer=feature_transformer,
        target_transformer=label_transformer
    )
    # call fit method
    aml_tx.fit(X, y)

    # return trained model
    return aml_tx


if __name__ == "__main__":
    """Entry point script that orchestrates the training of the data processing model.

    Usage
    -----

    $ python candidate_data_processors.train --processor_module dpp0
                                               [--data_dir /local/data]
                                               [--model_dir /local/path]
    """
    arg_parser = argparse.ArgumentParser(
        description='This is the entry point to start training '
        'the SageMaker Autopilot generated data processors.'
    )

    arg_parser.add_argument(
        '--processor_module',
        '-p',
        type=str,
        help='The data processors module to be executed.'
    )

    arg_parser.add_argument(
        '--data_dir',
        '-d',
        type=str,
        default='/opt/ml/input/data/train',
        help='Path to the directory containing training data.'
    )

    arg_parser.add_argument(
        '--model_dir',
        '-m',
        type=str,
        default='/opt/ml/model',
        help='Path to an existing directory where model will be saved.'
    )

    args, _ = arg_parser.parse_known_args()

    # load processor module
    processor = importlib.import_module(
        args.processor_module, package='candidate_data_processors'
    )

    # load header from processor module
    header = processor.HEADER

    # load feature processor from processor module
    feature_transformer = processor.build_feature_transform()

    # load label processor from processor module
    # absence of label processor implies that the labels are not processed
    try:
        label_transformer = processor.build_label_transform()
    except AttributeError:
        label_transformer = None

    X, y = read_csv_data(source=args.data_dir, target_column_index=header.target_column_index, output_dtype='O')
    logging.info('Feature data shape: {}'.format(X.shape))

    model = train(
        X, y,
        header=header,
        feature_transformer=feature_transformer,
        label_transformer=label_transformer
    )

    # serialize the model to model_dir
    dump(model, filename=os.path.join(args.model_dir, 'model.joblib'))

    # serialize the inference code to the model dir as well.
    serve_path = os.path.join(os.path.dirname(__file__), 'sagemaker_serve.py')
    shutil.copy(serve_path, args.model_dir)
    logging.info('Training completed, serializing to {}'.format(args.model_dir))
