import argparse
import logging
import json
from tensorflow.python.lib.io import file_io
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
import boto3
import tensorflow as tf
import datetime
from botocore.exceptions import ClientError


def main(argv=None):
    parser = argparse.ArgumentParser(description='Visualization exmaple')
    parser.add_argument('--s3_bucket', type=str, required=True, help='S3 Bucket to use.')
    parser.add_argument('--s3_prefix', type=str, help='S3 Bucket path prefix.', default="iris-example")

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
    iris = pd.read_csv(url, names=names)

    array = iris.values
    X, y = array[:, 0:4], np.where(array[:,4] == 'Iris-setosa', 1, 0)

    test_size=0.2
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    logging.info("Trained Model's evaluation score: {}".format(model.score(X_test, y_test)))

    df = pd.concat([pd.DataFrame(y_test, columns=['target']), pd.DataFrame(y_pred, columns=['predicted'])], axis=1)

    vocab = list(df['target'].unique())
    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)

    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = os.path.join('/tmp', 'confusion_matrix.csv')
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    time_hash = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "/tmp/logs/fit/" + time_hash
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(x=X_train,
              y=y_train,
              epochs=10,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback])


    # upload to S3
    AWS_REGION = 'us-west-2'
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        # upload cm file to S3
        cm_file_name = cm_file

        cm_object_name = 'confusion_matrix.csv'
        s3_cm_file='s3://' + args.s3_bucket + '/' + args.s3_prefix + '/' + cm_object_name
        cm_response = s3_client.upload_file(cm_file_name, args.s3_bucket, args.s3_prefix + '/' + cm_object_name)

        # upload tb log dir to S3
        s3_tb_file = 's3://' + args.s3_bucket + '/' + args.s3_prefix + '/tb-logs'
        for path, subdirs, files in os.walk(log_dir):
            path = path.replace("\\","/")
            directory_name = path.replace(log_dir, "")
            for file in files:
                s3_client.upload_file(os.path.join(path, file), args.s3_bucket, args.s3_prefix + '/tb-logs/'+directory_name+'/'+file)

    except ClientError as e:
        logging.info("ERROR IN S3 UPLOADING!!!!!!")
        logging.ERROR(e)

    logging.info("S3 object_name is: {}".format(s3_cm_file))

    metadata = {
        'outputs' : [
        # Markdown that is hardcoded inline
        {
            'storage': 'inline',
            'source': '# Inline Markdown\n[A link](https://www.kubeflow.org/)',
            'type': 'markdown',
        },
        {
            'source': 'https://raw.githubusercontent.com/kubeflow/pipelines/master/README.md',
            'type': 'markdown',
        },
        {
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': s3_cm_file,
            # Convert vocab to string because for bealean values we want "True|False" to match csv data.
            'labels': list(map(str, vocab)),
        },
        {
            'type': 'tensorboard',
            'source': s3_tb_file,
        }
        ]
    }

    with file_io.FileIO('/tmp/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    logging.info("Succeed in Markdown")


if __name__ == "__main__":
    main()
