import functools
import multiprocessing

from datetime import datetime
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "anaconda", "tensorflow==2.3.0", "-y"])
import tensorflow as tf
from tensorflow import keras

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "transformers==3.5.1", "-y"])
from transformers import DistilBertTokenizer
from transformers import DistilBertConfig

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])
import pandas as pd
import os
import re
import collections
import argparse
import json
import os
import numpy as np
import csv
import glob
from pathlib import Path
import tarfile
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

CLASSES = [1, 2, 3, 4, 5]

config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(CLASSES),
    id2label={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    label2id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
)


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
        "--input-model",
        type=str,
        default="/opt/ml/processing/input/model",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
    )

    return parser.parse_args()


def process(args):
    print("Current host: {}".format(args.current_host))

    print("input_data: {}".format(args.input_data))
    print("input_model: {}".format(args.input_model))

    print("Listing contents of input model dir: {}".format(args.input_model))
    input_files = os.listdir(args.input_model)
    for file in input_files:
        print(file)
    model_tar_path = "{}/model.tar.gz".format(args.input_model)
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(args.input_model)
    model_tar.close()

    model = keras.models.load_model("{}/tensorflow/saved_model/0".format(args.input_model))
    print(model)

    def predict(text):
        encode_plus_tokens = tokenizer.encode_plus(
            text, pad_to_max_length=True, max_length=args.max_seq_length, truncation=True, return_tensors="tf"
        )
        # The id from the pre-trained BERT vocabulary that represents the token.  (Padding of 0 will be used if the # of tokens is less than `max_seq_length`)
        input_ids = encode_plus_tokens["input_ids"]

        # Specifies which tokens BERT should pay attention to (0 or 1).  Padded `input_ids` will have 0 in each of these vector elements.
        input_mask = encode_plus_tokens["attention_mask"]

        outputs = model.predict(x=(input_ids, input_mask))

        prediction = [{"label": config.id2label[item.argmax()], "score": item.max().item()} for item in outputs]

        return prediction[0]["label"]

    print(
        """I loved it!  I will recommend this to everyone.""",
        predict("""I loved it!  I will recommend this to everyone."""),
    )

    print("""It's OK.""", predict("""It's OK."""))

    print(
        """Really bad.  I hope they don't make this anymore.""",
        predict("""Really bad.  I hope they don't make this anymore."""),
    )

    ###########################################################################################
    # TODO:  Replace this with glob for all files and remove test_data/ from the model.tar.gz #
    ###########################################################################################
    #    evaluation_data_path = '/opt/ml/processing/input/data/'

    print("Listing contents of input data dir: {}".format(args.input_data))
    input_files = os.listdir(args.input_data)

    test_data_path = "{}/amazon_reviews_us_Digital_Software_v1_00.tsv.gz".format(args.input_data)
    print("Using only {} to evaluate.".format(test_data_path))
    df_test_reviews = pd.read_csv(test_data_path, delimiter="\t", quoting=csv.QUOTE_NONE, compression="gzip")[
        ["review_body", "star_rating"]
    ]

    df_test_reviews = df_test_reviews.sample(n=100)
    df_test_reviews.shape
    df_test_reviews.head()

    y_test = df_test_reviews["review_body"].map(predict)
    y_test

    y_actual = df_test_reviews["star_rating"]
    y_actual

    print(classification_report(y_true=y_test, y_pred=y_actual))

    accuracy = accuracy_score(y_true=y_test, y_pred=y_actual)
    print("Test accuracy: ", accuracy)

    def plot_conf_mat(cm, classes, title, cmap):
        print(cm)
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="black" if cm[i, j] > thresh else "black",
            )

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

    cm = confusion_matrix(y_true=y_test, y_pred=y_actual)

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_conf_mat(cm, classes=CLASSES, title="Confusion Matrix", cmap=plt.cm.Greens)

    # Save the confusion matrix
    plt.show()

    # Model Output
    metrics_path = os.path.join(args.output_data, "metrics/")
    os.makedirs(metrics_path, exist_ok=True)
    plt.savefig("{}/confusion_matrix.png".format(metrics_path))

    report_dict = {
        "metrics": {
            "accuracy": {
                "value": accuracy,
            },
        },
    }

    evaluation_path = "{}/evaluation.json".format(metrics_path)
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    print("Listing contents of output dir: {}".format(args.output_data))
    output_files = os.listdir(args.output_data)
    for file in output_files:
        print(file)

    print("Listing contents of output/metrics dir: {}".format(metrics_path))
    output_files = os.listdir("{}".format(metrics_path))
    for file in output_files:
        print(file)

    print("Complete")


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
