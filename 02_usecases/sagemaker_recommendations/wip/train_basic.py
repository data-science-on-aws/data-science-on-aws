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

class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_embedding: tf.keras.Model,
      movie_embeddings: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_embeddings = user_embeddings
    self.movie_embeddings = movie_embeddings

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed using the retrieval task
    user_embeddings = self.user_embeddings(features['user_id'])
    movie_embeddings = self.movie_embeddings(features['movie_title'])

    return self.task(user_embeddings, movie_embeddings)

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
    use_xla = args.use_xla
    print('use_xla {}'.format(use_xla))
    use_amp = args.use_amp
    print('use_amp {}'.format(use_amp))
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
        'movie_title': x['movie_title'],
        'user_id': x['user_id']
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

    # Create the user vocabulary and user embeddings
    user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x: x['user_id']))

    user_embeddings = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(),
                                  embedding_dimension)
    ])

    # Create the movie vocabulary and movie embeddings
    movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)

    movie_embeddings = tf.keras.Sequential([
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(),
                                  embedding_dimension)
    ])

    # Specify the task and the top-k metric to optimize during model training
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_embeddings)
    ))

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

    # Create a custom Keras model with the user embeddings, movie embeddings, and optimization task
    model = MovieLensModel(user_embeddings, movie_embeddings, task)
    
    # Compile the model and prepare for training
    model.compile(optimizer=optimizer)

    # Train the model
    model.fit(ratings.batch(4096), epochs=epochs, callbacks=callbacks)

    # Make some sample predictions to test our model
    # Note:  This is required to save and server our model with TensorFlow Serving
    #        See https://github.com/tensorflow/tensorflow/issues/31057 for more  details.
    index = tfrs.layers.factorized_top_k.BruteForce(query_model=model.user_embeddings)
    index.index(movies.batch(100).map(model.movie_embeddings), movies)

    user_id = '42'
    _, titles = index(np.array([user_id]))

    k = 10
    print(f'Top {k} recommendations for user {user_id}: {titles[0, :k]}')

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