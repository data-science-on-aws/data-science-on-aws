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
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-recommenders==0.1.3'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-datasets==4.0.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib==3.2.1'])

from typing import Dict, Text
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

#     parser.add_argument('--train_data', 
#                         type=str, 
#                         default=os.environ['SM_CHANNEL_TRAIN'])
#     parser.add_argument('--validation_data', 
#                         type=str, 
#                         default=os.environ['SM_CHANNEL_VALIDATION'])
#     parser.add_argument('--test_data',
#                         type=str,
#                         default=os.environ['SM_CHANNEL_TEST'])
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
                        default=100)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.5)
    parser.add_argument('--enable_tensorboard',
                        type=eval,
                        default=False)        
    parser.add_argument('--output_data_dir', # This is unused
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    # This points to the S3 location - this should not be used by our code
    # We should use /opt/ml/model/ instead
    # parser.add_argument('--model_dir', 
    #                     type=str, 
    #                     default=os.environ['SM_MODEL_DIR'])
     
    args, _ = parser.parse_known_args()
    print("Args:") 
    print(args)
    
    env_var = os.environ 
    print("Environment Variables:") 
    pprint.pprint(dict(env_var), width = 1) 

    print('SM_TRAINING_ENV {}'.format(env_var['SM_TRAINING_ENV']))
    sm_training_env_json = json.loads(env_var['SM_TRAINING_ENV'])
    is_master = sm_training_env_json['is_master']
    print('is_master {}'.format(is_master))
    
#     train_data = args.train_data
#     print('train_data {}'.format(train_data))
#     validation_data = args.validation_data
#     print('validation_data {}'.format(validation_data))
#     test_data = args.test_data
#     print('test_data {}'.format(test_data))    

    local_model_dir = os.environ['SM_MODEL_DIR']
    output_dir = args.output_dir
    print('output_dir {}'.format(output_dir))    
    hosts = args.hosts
    print('hosts {}'.format(hosts))    
    current_host = args.current_host
    print('current_host {}'.format(current_host))    
    num_gpus = args.num_gpus
    print('num_gpus {}'.format(num_gpus))
    job_name = os.environ['SAGEMAKER_JOB_NAME']
    print('job_name {}'.format(job_name))    

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

    # Determine if PipeMode is enabled 
#     pipe_mode_str = os.environ.get('SM_INPUT_DATA_CONFIG', '')
#     pipe_mode = (pipe_mode_str.find('Pipe') >= 0)
#     print('Using pipe_mode: {}'.format(pipe_mode))
 
    # SavedModel Output
    tensorflow_saved_model_path = os.path.join(local_model_dir, 'tensorflow/saved_model/0')
    os.makedirs(tensorflow_saved_model_path, exist_ok=True)

    # Tensorboard Logs 
    tensorboard_logs_path = os.path.join(local_model_dir, 'tensorboard/')
    os.makedirs(tensorboard_logs_path, exist_ok=True)

    # Commented out due to incompatibility with transformers library (possibly)
    # Set the global precision mixed_precision policy to "mixed_float16"    
#    mixed_precision_policy = 'mixed_float16'
#    print('Mixed precision policy {}'.format(mixed_precision_policy))
#    policy = mixed_precision.Policy(mixed_precision_policy)
#    mixed_precision.set_policy(policy)    
    
    from typing import Dict, Text

    import numpy as np
    import tensorflow as tf

    import tensorflow_datasets as tfds
    import tensorflow_recommenders as tfrs

    distributed_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with distributed_strategy.scope():
        tf.config.optimizer.set_jit(use_xla)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": use_amp})

#         train_data_filenames = glob(os.path.join(train_data, '*.tfrecord'))
#         print('train_data_filenames {}'.format(train_data_filenames))
#         train_dataset = file_based_input_dataset_builder(
#             channel='train',
#             input_filenames=train_data_filenames,
#             pipe_mode=pipe_mode,
#             is_training=True,
#             drop_remainder=False,
#             batch_size=train_batch_size,
#             epochs=epochs,
#             steps_per_epoch=train_steps_per_epoch,
#             max_seq_length=max_seq_length).map(select_data_and_label_from_record)

        ratings = tfds.load('movielens/100k-ratings', split="train")
        print(ratings)

        movies = tfds.load('movielens/100k-movies', split="train")
        print(movies)

        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"]
        })
        print(ratings)

        movies = movies.map(lambda x: x["movie_title"])
        print(movies)

        user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
        user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

        movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
        movie_titles_vocabulary.adapt(movies)

        user_model = tf.keras.Sequential([
            user_ids_vocabulary,
            tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 128)
        ])

        movie_model = tf.keras.Sequential([
            movie_titles_vocabulary,
            tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 128)
        ])

        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            movies.batch(128).map(movie_model)
          )
        )        

        optimizer = tf.keras.optimizers.Adagrad(learning_rate)
        print('** use_amp {}'.format(use_amp))        
        if use_amp:
            # loss scaling is currently required when using mixed precision
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')

        callbacks = []
        
        if enable_tensorboard:            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                        log_dir=tensorboard_logs_path)
            print('*** TENSORBOARD CALLBACK {} ***'.format(tensorboard_callback))
            callbacks.append(tensorboard_callback)
  
        print('*** OPTIMIZER {} ***'.format(optimizer))
        
        model = MovieLensModel(user_model, movie_model, task)          
        model.compile(optimizer=optimizer)

        model.fit(ratings.batch(4096), epochs=epochs)

        print('Compiled model {}'.format(model))          
        print(model.summary())

        index = tfrs.layers.ann.BruteForce(model.user_model)
        index.index(movies.batch(100).map(model.movie_model), movies)

        # Get some recommendations.
        _, titles = index(np.array(["42"]))
        print(f"Top 10 recommendations for user 42: {titles[0, :10]}")

        # Save the TensorFlow SavedModel for Serving Predictions
        # Note:  We must call index() above before we save().
        #        See https://github.com/tensorflow/tensorflow/issues/31057 for more details.
        print('tensorflow_saved_model_path {}'.format(tensorflow_saved_model_path))   
        index.save(tensorflow_saved_model_path, save_format='tf')
                
        # Copy inference.py and requirements.txt to the code/ directory
        #   Note: This is required for the SageMaker Endpoint to pick them up.
        #         This appears to be hard-coded and must be called code/
        inference_path = os.path.join(local_model_dir, 'code/')
        print('Copying inference source files to {}'.format(inference_path))
        os.makedirs(inference_path, exist_ok=True)               
        os.system('cp inference.py {}'.format(inference_path))
        print(glob(inference_path))        
#        os.system('cp requirements.txt {}/code'.format(inference_path))
        

    