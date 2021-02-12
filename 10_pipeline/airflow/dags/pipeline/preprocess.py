import pandas as pd
import numpy as np
import s3fs


def preprocess(s3_in_url, s3_out_bucket, s3_out_prefix, delimiter=","):
    """Preprocesses data based on business logic

    - Reads delimited file passed as s3_url and preprocess data by filtering
    long tail in the customer ratings data i.e. keep customers who have rated 5
    or more videos, and videos that have been rated by 9+ customers
    - Preprocessed data is then written to output

    Args:
        s3_in_url:
          s3 url to the delimited file to be processed
          e.g. s3://amazon-reviews-pds/tsv/reviews.tsv.gz
        s3_out_bucket:
          s3 bucket where preprocessed data will be staged
          e.g. mybucket
        s3_out_prefix:
          s3 url prefix to stage preprocessed data to use later in the pipeline
          e.g. amazon-reviews-pds/preprocess/
        delimiter:
          delimiter to be used for parsing the file. Defaults to "," if none
          provided

    Returns:
        status of preprocessed data

    Raises:
        IOError: An error occurred accessing the s3 file
    """
    try:
        print("preprocessing data from {}".format(s3_in_url))
        # read s3 url into pandas dataframe
        # pandas internally uses s3fs to read s3 file directory
        df = pd.read_csv(s3_in_url, delimiter, error_bad_lines=False)

        # limit dataframe to customer_id, product_id, and star_rating
        # `product_title` will be useful validating recommendations
        df = df[["customer_id", "product_id", "star_rating", "product_title"]]

        # clean out the long tail because most people haven't seen most videos,
        # and people rate fewer videos than they actually watch
        customers = df["customer_id"].value_counts()
        products = df["product_id"].value_counts()

        # based on data exploration only about 5% of customers have rated 5 or
        # more videos, and only 25% of videos have been rated by 9+ customers
        customers = customers[customers >= 5]
        products = products[products >= 10]
        print("# of rows before the long tail = {:10d}".format(df.shape[0]))
        reduced_df = df.merge(pd.DataFrame({"customer_id": customers.index})).merge(
            pd.DataFrame({"product_id": products.index})
        )
        print("# of rows after the long tail = {:10d}".format(reduced_df.shape[0]))
        reduced_df = reduced_df.drop_duplicates(["customer_id", "product_id"])
        print("# of rows after removing duplicates = {:10d}".format(reduced_df.shape[0]))

        # recreate customer and product lists since there are customers with
        # more than 5 reviews, but all of their reviews are on products with
        # less than 5 reviews (and vice versa)
        customers = reduced_df["customer_id"].value_counts()
        products = reduced_df["product_id"].value_counts()

        # sequentially index each user and item to hold the sparse format where
        # the indices indicate the row and column in our ratings matrix
        customer_index = pd.DataFrame({"customer_id": customers.index, "customer": np.arange(customers.shape[0])})
        product_index = pd.DataFrame({"product_id": products.index, "product": np.arange(products.shape[0])})
        reduced_df = reduced_df.merge(customer_index).merge(product_index)

        nb_customer = reduced_df["customer"].max() + 1
        nb_products = reduced_df["product"].max() + 1
        feature_dim = nb_customer + nb_products
        print(nb_customer, nb_products, feature_dim)

        product_df = reduced_df[["customer", "product", "star_rating"]]

        # split into train, validation and test data sets
        train_df, validate_df, test_df = np.split(
            product_df.sample(frac=1), [int(0.6 * len(product_df)), int(0.8 * len(product_df))]
        )

        print("# of rows train data set = {:10d}".format(train_df.shape[0]))
        print("# of rows validation data set = {:10d}".format(validate_df.shape[0]))
        print("# of rows test data set = {:10d}".format(test_df.shape[0]))

        # select columns required for training the model
        # excluding columns "customer_id", "product_id", "product_title" to
        # keep files small
        cols = ["customer", "product", "star_rating"]
        train_df = train_df[cols]
        validate_df = validate_df[cols]
        test_df = test_df[cols]

        # write output to s3 as delimited file
        fs = s3fs.S3FileSystem(anon=False)
        s3_out_prefix = s3_out_prefix[:-1] if s3_out_prefix[-1] == "/" else s3_out_prefix
        s3_out_train = "s3://{}/{}/{}".format(s3_out_bucket, s3_out_prefix, "train/train.csv")
        print("writing training data to {}".format(s3_out_train))
        with fs.open(s3_out_train, "w") as f:
            train_df.to_csv(f, sep=str(","), index=False)

        s3_out_validate = "s3://{}/{}/{}".format(s3_out_bucket, s3_out_prefix, "validate/validate.csv")
        print("writing test data to {}".format(s3_out_validate))
        with fs.open(s3_out_validate, "w") as f:
            validate_df.to_csv(f, sep=str(","), index=False)

        s3_out_test = "s3://{}/{}/{}".format(s3_out_bucket, s3_out_prefix, "test/test.csv")
        print("writing test data to {}".format(s3_out_test))
        with fs.open(s3_out_test, "w") as f:
            test_df.to_csv(f, sep=str(","), index=False)

        print("preprocessing completed")
        return "SUCCESS"
    except Exception as e:
        raise e
