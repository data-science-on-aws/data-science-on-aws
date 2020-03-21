import argparse, os
import numpy as np
import pandas as pd

import tensorflow as tf
import keras

import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
if __name__ == '__main__':      
    
    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dense-layer', type=int, default=32)
    parser.add_argument('--layers', type=float, default=2)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()
    epochs        = args.epochs
    learning_rate = args.learning_rate
    batch_size    = args.batch_size
    dense_layer   = args.dense_layer
    layers        = args.layers
    gpu_count     = args.gpu_count
    model_dir     = args.model_dir
    training_dir  = args.training
    
    # Read data set
    data = pd.read_csv(training_dir+'/bank-additional-full.csv', sep=';')
    print("Data shape: ", data.shape)
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data)
    print("One-hot encoded data shape: ", data.shape)

    # Separate features and labels
    X = data.drop(['y_yes', 'y_no'], axis=1)
    Y = data['y_yes']
    print("X shape: ", data.shape)
    print("Y shape: ", data.shape)

    # Scale numerical features
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X)
    
    # Split data set for training and validation
    from sklearn import model_selection
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(X, Y, test_size=0.1, random_state=123)

    # Number of features in the training set (we need to pass this value to the input layer)
    input_dim = x_train.shape[1]

    # Build simple neural network
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(dense_layer, input_dim=input_dim, activation='relu'))
    for i in range(int(layers)-1):
        model.add(Dense(dense_layer, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # because we want a probability between 0 and 1
                                              # https://en.wikipedia.org/wiki/Sigmoid_function

    from keras.optimizers import SGD
    sgd = SGD(lr=learning_rate)
    
    import keras_metrics
    model.compile(loss='binary_crossentropy', optimizer=sgd, 
              metrics=['binary_accuracy', 
                   keras_metrics.precision(), 
                   keras_metrics.recall(),
                   keras_metrics.f1_score()])

    print(model.summary())

    # Train
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), 
          epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    score = model.evaluate(x_validation, y_validation, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    
    # save Keras model for Tensorflow Serving
    import keras.backend as K
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})


