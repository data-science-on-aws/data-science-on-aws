# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import argparse
import os
import numpy as np
import json
from datetime import datetime

class SyncToS3(tf.keras.callbacks.Callback):
    def __init__(self, logdir, s3logdir):
        super(SyncToS3, self).__init__()
        self.logdir = logdir
        self.s3logdir = s3logdir
    
    # Explicitly sync to S3 upon completion
    def on_epoch_end(self, batch, logs={}):
        os.system('aws s3 sync ' + self.logdir + ' ' + self.s3logdir)
        # ' >/dev/null 2>&1'

def model(x_train, y_train, x_test, y_test, args):
    """Generate a simple model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=args.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = []
    logdir = args.output_data_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
    callbacks.append(TensorBoard(log_dir=logdir, profile_batch=0))
    callbacks.append(SyncToS3(logdir=logdir, s3logdir=args.model_dir))
    
    model.fit(x=x_train, 
              y=y_train,
              callbacks=callbacks,
              epochs=args.epochs)

    score = model.evaluate(x=x_test, 
                           y=y_test)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])
    
    return model


def _load_training_data(base_dir):
    """Load MNIST training data"""
    print(base_dir)
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
    print(base_dir)
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--epochs',           type=int,   default=5)
    parser.add_argument('--optimizer',        type=str,   default='adam')

    # SageMaker parameters
    # model_dir is always passed in from SageMaker. (By default, it is a S3 path under the default bucket.)
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--sm-model-dir',     type=str,   default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_output_dir', type=str,   default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_data_dir',  type=str,   default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    # Data directories and other options
    parser.add_argument('--train',            type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--hosts',            type=list,  default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host',     type=str,   default=os.environ['SM_CURRENT_HOST'])

    return parser.parse_known_args()


if __name__ == "__main__":
    
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    mnist_classifier = model(train_data, 
                             train_labels, 
                             eval_data, 
                             eval_labels, 
                             args)

    print('current_host:  {}'.format(args.current_host))
    print('hosts[0]:  {}'.format(args.hosts[0]))
    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000001'), './sm_tensorflow_mnist.h5')
        mnist_classifier.save(os.path.join('/opt/ml/model/', '000000001'), './opt_tensorflow_mnist.h5')        
        
        # TODO:  Copy .h5 file to /opt/ml/model/ (backed by S3)
        # import shutil
        # shutil.copyfile('./sm_tensorflow_mnist.h5', '/opt/ml/model/000000001/sm_tensorflow_mnist.h5')
        # shutil.copyfile('./opt_tensorflow_mnist.h5', '/opt/ml/model/000000001/opt_tensorflow_mnist.h5')