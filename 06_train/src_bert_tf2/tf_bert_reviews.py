import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.2.0-rc1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bert-for-tf2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentencepiece'])

import tensorflow as tf
print(tf.__version__)

import boto3
import sagemaker
import pandas as pd

sess   = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

sm = boto3.Session().client(service_name='sagemaker', region_name=region)

import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from sklearn.metrics import confusion_matrix, classification_report

train = pd.read_csv('./data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', delimiter='\t')[:100]
test = pd.read_csv('./data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', delimiter='\t')[:100]

train.shape
train.head()

import os

os.system('rm uncased_L-12_H-768_A-12.zip')
os.system('rm -rf uncased_L-12_H-768_A-12')

os.system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')

os.system('unzip uncased_L-12_H-768_A-12.zip')
os.system('ls -al ./uncased_L-12_H-768_A-12')
#subprocess.check_call([sys.executable, 'unzip', '-f', 'uncased_L-12_H-768_A-12.zip'])
#subprocess.check_call([sys.executable, 'ls', '-al', './model/uncased_L-12_H-768_A-12'])

bert_ckpt_dir = './uncased_L-12_H-768_A-12'
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

class ClassificationData:
  TEXT_COLUMN = 'review_body'
  LABEL_COLUMN = 'star_rating'

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

#    print('max seq_len', self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[ClassificationData.TEXT_COLUMN], row[ClassificationData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

tokenizer.tokenize("I can't wait to visit Bulgaria again!")

tokens = tokenizer.tokenize("I can't wait to visit Bulgaria again!")
tokenizer.convert_tokens_to_ids(tokens)


def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)
        
  return model



import numpy as np

classes = train['star_rating'].unique().tolist()
classes

features = ClassificationData(train, test, tokenizer, classes, max_seq_len=128)

features.train_x.shape

features.train_x[0]

features.train_y[0]

features.max_seq_len

model = create_model(features.max_seq_len, bert_ckpt_file)

model.summary()

model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

log_dir = "log/classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
  x=features.train_x, 
  y=features.train_y,
  validation_split=0.1,
  batch_size=8,
  shuffle=False,
  epochs=1,
  callbacks=[tensorboard_callback]
)

_, train_acc = model.evaluate(features.train_x, features.train_y)
_, test_acc = model.evaluate(features.test_x, features.test_y)

print("train acc", train_acc)
print("test acc", test_acc)

y_pred = model.predict(features.test_x).argmax(axis=-1)


print(classification_report(features.test_y, y_pred)) #, target_names=classes))

cm = confusion_matrix(features.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)

sentences = [
  "This is just OK.",
  "This sucks.",
  "This is great."
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(features.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for review_body, star_rating in zip(sentences, predictions):
  print("review_body:", review_body, "\star_rating:", classes[star_rating])
  print()

