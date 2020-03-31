import sys
import subprocess
import argparse
import json
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-gpu==2.2.0-rc2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bert-for-tf2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentencepiece'])

import tensorflow as tf
print(tf.__version__)

import boto3
import pandas as pd

import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from glob import glob 

from bert.model import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from sklearn.metrics import confusion_matrix, classification_report

#train = pd.read_csv('./data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', delimiter='\t')[:100]
#test = pd.read_csv('./data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', delimiter='\t')[:100]

#train.shape
#train.head()

import os

os.system('rm uncased_L-12_H-768_A-12.zip')
os.system('rm -rf uncased_L-12_H-768_A-12')

os.system('wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')

#os.system('unzip uncased_L-12_H-768_A-12.zip')

import zipfile
with zipfile.ZipFile('uncased_L-12_H-768_A-12.zip', 'r') as zip_ref:
  zip_ref.extractall('.')


os.system('ls -al ./uncased_L-12_H-768_A-12')
#subprocess.check_call([sys.executable, 'unzip', '-f', 'uncased_L-12_H-768_A-12.zip'])
#subprocess.check_call([sys.executable, 'ls', '-al', './model/uncased_L-12_H-768_A-12'])

bert_ckpt_dir = './uncased_L-12_H-768_A-12'
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

CLASSES=[1, 2, 3, 4, 5]
MAX_SEQ_LEN=128
BATCH_SIZE=128
EPOCHS=2
STEPS_PER_EPOCH=1000

def select_data_and_label_from_record(record):
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']

    return (x, y)


def file_based_input_dataset_builder(input_file, 
                                     seq_length, 
                                     is_training,
                                     drop_remainder):

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

#  def input_fn(params):
#  """The actual input function."""
#  batch_size = params["batch_size"]

  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  dataset = tf.data.TFRecordDataset(input_file)
  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100)

  dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
          lambda record: _decode_record(record, name_to_features),
          batch_size=BATCH_SIZE,
          drop_remainder=drop_remainder))

  return dataset

#  return input_fn


# class ClassificationData:
#   TEXT_COLUMN = 'review_body'
#   LABEL_COLUMN = 'star_rating'

#   def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
#     self.tokenizer = tokenizer
#     self.max_seq_len = 0
#     self.classes = classes
    
#     ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

# #    print('max seq_len', self.max_seq_len)
#     self.max_seq_len = min(self.max_seq_len, max_seq_len)
#     self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

#   def _prepare(self, df):
#     x, y = [], []
    
#     for _, row in tqdm(df.iterrows()):
#       text, label = row[ClassificationData.TEXT_COLUMN], row[ClassificationData.LABEL_COLUMN]
#       tokens = self.tokenizer.tokenize(text)
#       tokens = ["[CLS]"] + tokens + ["[SEP]"]
#       token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#       self.max_seq_len = max(self.max_seq_len, len(token_ids))
#       x.append(token_ids)
#       y.append(self.classes.index(label))

#     return np.array(x), np.array(y)

#   def _pad(self, ids):
#     x = []
#     for input_ids in ids:
#       input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
#       input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
#       x.append(np.array(input_ids))
#     return np.array(x)

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

tokenizer.tokenize("I can't wait to visit Bulgaria again!")

tokens = tokenizer.tokenize("I can't wait to visit Bulgaria again!")
tokenizer.convert_tokens_to_ids(tokens)


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


def create_model(max_seq_len, bert_ckpt_file, adapter_size):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
    bc = StockBertConfig.from_json_string(reader.read())
    bert_params = map_stock_config_to_params(bc)
    bert_params.adapter_size = adapter_size 
    bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=len(CLASSES), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)

  if adapter_size is not None:
    freeze_bert_layers(bert)

  return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

#    parser.add_argument('--model-type', type=str, default='bert')
#    parser.add_argument('--model-name', type=str, default='bert-base-cased')
    parser.add_argument('--train-data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args, _ = parser.parse_known_args()   
#    model_type = args.model_type
#    model_name = args.model_name
    train_data = args.train_data
    validation_data = args.validation_data
    model_dir = args.model_dir
    hosts = args.hosts
    current_host = args.current_host
    num_gpus = args.num_gpus

    # features = ClassificationData(train, test, tokenizer, classes, max_seq_len=128)
    # features.train_x.shape
    # features.train_x[0]
    # features.train_y[0]
    # features.max_seq_len

    adapter_size = None # Change to 64?
    model = create_model(MAX_SEQ_LEN, bert_ckpt_file, adapter_size)

    model.summary()

    model.compile(
      optimizer=keras.optimizers.Adam(1e-5),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    log_dir = "log/classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    train_data_filenames = glob('{}/*.tfrecord'.format(train_data))
    print(train_data_filenames)

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_dataset = file_based_input_dataset_builder(
        train_data_filenames,
        seq_length=MAX_SEQ_LEN,
        is_training=True,
        drop_remainder=False)

    print('*********** {}'.format(train_dataset))

#    (train_dataset_X, train_dataset_y) = train_dataset.map(select_data_and_label_from_record)
    train_dataset_2 = train_dataset.map(select_data_and_label_from_record)
    print(train_dataset_2)

    iterator = iter(train_dataset_2)
    next_element = iterator.get_next()
    print('*********** {}'.format(next_element))

#    if is_training:
#        dataset = dataset.shuffle(100)
#        dataset = dataset.repeat()

#    dataset = dataset.batch(batch_size, drop_remainder=is_training)
#    dataset = dataset.prefetch(1024)

#    iterator = iter(train_dataset_2)
#    next_element = iterator.get_next()
#    print('*********** {}'.format(next_element))

    history = model.fit(
      train_dataset_2,
#       x=train_dataset_X,
#       y=train_dataset_y,
#      train_dataset.batch(10),
#      train_dataset,
#      x=features.train_x, 
#      y=features.train_y,
#      validation_split=0.1,
      batch_size=BATCH_SIZE,
      shuffle=True,
      epochs=EPOCHS,
      steps_per_epoch=STEPS_PER_EPOCH,
      callbacks=[tensorboard_callback]
    )

#    _, train_acc = model.evaluate(features.train_x, features.train_y)
#    _, test_acc = model.evaluate(features.test_x, features.test_y)

#    print("train acc", train_acc)
#    print("test acc", test_acc)

#    y_pred = model.predict(features.test_x).argmax(axis=-1)

#    print(classification_report(features.test_y, y_pred)) #, target_names=classes))

#    cm = confusion_matrix(features.test_y, y_pred)
#    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    sentences = [
      "This is just OK.",
      "This sucks.",
      "This is great."
    ]

    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(MAX_SEQ_LEN-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    predictions = model.predict(pred_token_ids).argmax(axis=-1)

    for review_body, star_rating in zip(sentences, predictions):
       print("review_body:", review_body, "\star_rating:", CLASSES[star_rating])
       print()

#    model.save('/opt/ml/model/0/', save_format='tf')
#    model.save('/opt/ml/model/bert_reviews.h5')
