import time
import random
import pandas as pd
from glob import glob
import pprint
import argparse
import json
import subprocess
import sys
import os

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn==0.23.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-tensorflow==2.3.0.1.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-recommenders==0.2.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-datasets==4.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib==3.2.1'])
        
from typing import Dict, Text
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import numpy as np


class MovielensModel(tfrs.models.Model):

  def __init__(self, embedding_dimension: int, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

#    embedding_dimension = 32

    # User and movie models.
    self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.movie_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.movie_model(features["movie_title"])
    
    return (
        user_embeddings,
        movie_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    user_embeddings, movie_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=features["user_rating"],
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)


if __name__ == '__main__':    
    env_var = os.environ 
    print('Environment Variables:') 
    pprint.pprint(dict(env_var), width = 1) 
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--hosts', 
                        type=list, 
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', 
                        type=str, 
                        default=os.environ['SM_CURRENT_HOST'])    
    parser.add_argument('--num_gpus', 
                        type=int, 
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--use_xla',
                        type=eval,
                        default=False)
    parser.add_argument('--use_amp',
                        type=eval,
                        default=False)
    parser.add_argument('--epochs',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.5)
    parser.add_argument('--enable_tensorboard',
                        type=eval,
                        default=False)        
    parser.add_argument('--output_data_dir', # This is unused
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--dataset_variant', 
                        type=str, 
                        default='100k')
    parser.add_argument('--embedding_dimension', 
                        type=str, 
                        default=256)
    
    args, _ = parser.parse_known_args()
    print('command line args:') 
    print(args)
    train_data = args.train_data
    print('train_data {}'.format(train_data))
    local_model_dir = os.environ['SM_MODEL_DIR']
    print('local_model_dir {}'.format(local_model_dir))
    output_dir = args.output_dir
    print('output_dir {}'.format(output_dir))    
    hosts = args.hosts
    print('hosts {}'.format(hosts))    
    current_host = args.current_host
    print('current_host {}'.format(current_host))    
    num_gpus = args.num_gpus
    print('num_gpus {}'.format(num_gpus))
    epochs = args.epochs
    print('epochs {}'.format(epochs))
    learning_rate = args.learning_rate
    print('learning_rate {}'.format(learning_rate))
    enable_tensorboard = args.enable_tensorboard
    print('enable_tensorboard {}'.format(enable_tensorboard))
    dataset_variant = args.dataset_variant
    print('dataset_variant {}'.format(dataset_variant))
    embedding_dimension = int(args.embedding_dimension)
    print('embedding_dimension {}'.format(embedding_dimension))    
             
    # Load the ratings data to use for training
    ratings = tfds.load('movielens/{}-ratings'.format(dataset_variant), 
                        download=False,
                        data_dir=train_data,
                        split='train')
    print('Ratings raw', ratings)

    # Transform the ratings data specific to our training task
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })
    print('Ratings transformed', ratings)    

    # Load the movies data to use for training
    movies = tfds.load('movielens/{}-movies'.format(dataset_variant),
                       download=False,
                       data_dir=train_data,
                       split='train')
    print('Movies raw', movies)
    
    # Transform the movies data specific to our training task
    movies = movies.map(lambda x: x['movie_title'])
    print('Movies transformed', movies)

    # Randomly shuffle data and split between train and test.
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()
    
    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(100_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    # Define the optimizer and hyper-parameters
    optimizer = tf.keras.optimizers.Adagrad(learning_rate)
    print('Optimizer:  {}'.format(optimizer))

    # Setup the callbacks to use during training
    callbacks = []

    # Setup the Tensorboard callback if Tensorboard is enabled
    if enable_tensorboard: 
        # Tensorboard Logs 
        tensorboard_logs_path = os.path.join(local_model_dir, 'tensorboard/')
        os.makedirs(tensorboard_logs_path, exist_ok=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
        print('Adding Tensorboard callback {}'.format(tensorboard_callback))
        callbacks.append(tensorboard_callback)
    print('Callbacks: {}'.format(callbacks))

    # Create and compile a custom Keras model specialized for rating
    model = MovielensModel(embedding_dimension=embedding_dimension, 
                           rating_weight=1.0, 
                           retrieval_weight=0.0)
    model.compile(optimizer=optimizer)

    # Train the model
    model.fit(cached_train, epochs=epochs, callbacks=callbacks)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"retrieval-top-100-accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"ranking-rmse: {metrics['root_mean_squared_error']:.3f}.")

    # Create and compile a custom Keras model specialized for retrieval
    model = MovielensModel(embedding_dimension=embedding_dimension, 
                           rating_weight=0.0, 
                           retrieval_weight=1.0)   
    model.compile(optimizer=optimizer)

    model.fit(cached_train, epochs=epochs, callbacks=callbacks)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"retrieval-top-100-accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"ranking-rmse: {metrics['root_mean_squared_error']:.3f}.")
    
    # Create and compile a custom Keras model for both rating and retrieval
    model = MovielensModel(embedding_dimension=embedding_dimension, 
                           rating_weight=1.0, 
                           retrieval_weight=1.0)
    model.compile(optimizer=optimizer)
    
    model.fit(cached_train, epochs=epochs, callbacks=callbacks)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")    
    
    # Make some sample predictions to test our model
    # Note:  This is required to save and server our model with TensorFlow Serving
    #        See https://github.com/tensorflow/tensorflow/issues/31057 for more  details.
# Create a model that takes in raw query features, and returns the predicted movie titles
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index(movies.batch(100).map(model.movie_model), movies)

    k = 5
    user_id = "42"

    _, titles = index(np.array([user_id]))

    print(f"Top {k} recommendations for user {user_id}: {titles[0, :k]}")

    # Print a summary of our recommender model
    print('Trained index {}'.format(index))
    print(index.summary())

    # Save the TensorFlow SavedModel for Serving Predictions
    # SavedModel Output
    tensorflow_saved_model_path = os.path.join(local_model_dir,
                                               'tensorflow/saved_model/0')
    os.makedirs(tensorflow_saved_model_path, exist_ok=True)
    
    print('tensorflow_saved_model_path {}'.format(tensorflow_saved_model_path))
    index.save(tensorflow_saved_model_path, save_format='tf')

    # Copy inference.py and requirements.txt to the code/ directory
    #   Note: This is required for the SageMaker Endpoint to pick them up.
    #         This directory must be named `code/`
    inference_path = os.path.join(local_model_dir, 'code/')
    print('Copying inference.py to {}'.format(inference_path))
    os.makedirs(inference_path, exist_ok=True)               
    os.system('cp inference.py {}'.format(inference_path))
    print(glob(inference_path))