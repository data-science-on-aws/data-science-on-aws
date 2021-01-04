import sagemaker.amazon.common as smac
import pandas as pd
import numpy as np
import boto3
import s3fs
import io

from scipy.sparse import lil_matrix


def convert_sparse_matrix(df, nb_rows, nb_customer, nb_products):
    # dataframe to array
    df_val = df.values

    # determine feature size
    nb_cols = nb_customer + nb_products
    print("# of rows = {}".format(str(nb_rows)))
    print("# of cols = {}".format(str(nb_cols)))

    # extract customers and ratings
    df_X = df_val[:, 0:2]
    # Features are one-hot encoded in a sparse matrix
    X = lil_matrix((nb_rows, nb_cols)).astype('float32')
    df_X[:, 1] = nb_customer + df_X[:, 1]
    coords = df_X[:, 0:2]
    X[np.arange(nb_rows), coords[:, 0]] = 1
    X[np.arange(nb_rows), coords[:, 1]] = 1

    # create label with ratings
    Y = df_val[:, 2].astype('float32')

    # validate size and shape
    print(X.shape)
    print(Y.shape)
    assert X.shape == (nb_rows, nb_cols)
    assert Y.shape == (nb_rows, )

    return X, Y


def save_as_protobuf(X, Y, bucket, key):
    """Converts features and predictions matrices to recordio protobuf and
       writes to S3

    Args:
        X:
          2D numpy matrix with features
        Y:
          1D numpy matrix with predictions
        bucket:
          s3 bucket where recordio protobuf file will be staged
        prefix:
          s3 url prefix to stage prepared data to use for training the model
        key:
          protobuf file name to be staged

    Returns:
        s3 url with key to the protobuf data
    """
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, Y)
    buf.seek(0)
    obj = '{}'.format(key)
    boto3.resource('s3').Bucket(bucket).Object(obj).upload_fileobj(buf)
    return 's3://{}/{}'.format(bucket, obj)


def chunk(x, batch_size):
    """split array into chunks of batch_size
    """
    chunk_range = range(0, x.shape[0], batch_size)
    chunks = [x[p: p + batch_size] for p in chunk_range]
    return chunks


def prepare(s3_in_bucket,
            s3_in_prefix,
            s3_out_bucket,
            s3_out_prefix,
            delimiter=","):
    """Prepare data for training with Sagemaker algorithms

    - Read preprocessed data and converts to ProtoBuf format to prepare for
      training with Sagemaker algorithms

    Args:
        s3_in_bucket:
          s3 bucket where preprocessed files are staged
          e.g. mybucket
        s3_in_prefix:
          s3 prefix to the files to be used for training
          e.g. amazon-reviews-pds/preprocess/
          it's expected to have train and test folders in this prefix that will
          be staged by preprocessor
        s3_out_bucket:
          s3 bucket where training and test files will be staged
          e.g. mybucket
        s3_out_prefix:
          s3 url prefix to stage prepared data to use for training the model
          e.g. amazon-reviews-pds/prepare/
        delimiter:
          delimiter to be used for parsing the file. Defaults to "," if none
          provided

    Returns:
        s3 url with key to the prepared data

    Raises:
        IOError: An error occurred accessing the s3 file
    """
    try:
        print("preparing data from {}".format(s3_in_prefix))

        # prepare training data set
        if s3_in_prefix[-1] == "/":
            s3_in_prefix = s3_in_prefix[:-1]
        s3_train_url = "s3://{}/{}/{}".format(
            s3_in_bucket, s3_in_prefix, 'train/train.csv')
        train_df = pd.read_csv(s3_train_url,
                               sep=str(','), error_bad_lines=False)

        # prepare validateion dataset
        s3_validate_url = "s3://{}/{}/{}".format(
            s3_in_bucket, s3_in_prefix, 'validate/validate.csv')
        validate_df = pd.read_csv(s3_validate_url,
                                  sep=str(','), error_bad_lines=False)

        # prepare test dataset
        s3_test_url = "s3://{}/{}/{}".format(
            s3_in_bucket, s3_in_prefix, 'test/test.csv')
        test_df = pd.read_csv(s3_test_url,
                              sep=str(','), error_bad_lines=False)

        # get feature dimension
        all_df = pd.concat([train_df, validate_df, test_df])
        nb_customer = np.unique(all_df['customer'].values).shape[0]
        nb_products = np.unique(all_df['product'].values).shape[0]
        feature_dim = nb_customer + nb_products
        print(nb_customer, nb_products, feature_dim)

        train_X, train_Y = convert_sparse_matrix(
            train_df, train_df.shape[0], nb_customer, nb_products)
        validate_X, validate_Y = convert_sparse_matrix(
            validate_df, validate_df.shape[0], nb_customer, nb_products)
        test_X, test_Y = convert_sparse_matrix(
            test_df, test_df.shape[0], nb_customer, nb_products)

        # write train and test in protobuf format to s3
        if s3_out_prefix[-1] == "/":
            s3_out_prefix = s3_out_prefix[:-1]
        train_data = save_as_protobuf(
            train_X, train_Y, s3_out_bucket,
            s3_out_prefix + "/" + "train/train.protobuf")
        print(train_data)
        validate_data = save_as_protobuf(
            validate_X, validate_Y, s3_out_bucket,
            s3_out_prefix + "/" + "validate/validate.protobuf")
        print(validate_data)

        # chunk test data to avoid payload size issues when batch transforming
        test_x_chunks = chunk(test_X, 10000)
        test_y_chunks = chunk(test_Y, 10000)
        N = len(test_x_chunks)
        for i in range(N):
            test_data = save_as_protobuf(
                test_x_chunks[i],
                test_y_chunks[i],
                s3_out_bucket,
                s3_out_prefix + "/" + "test/test_" + str(i) + ".protobuf")
            print(test_data)

        return "SUCCESS"
    except Exception as e:
        raise e
