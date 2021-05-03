#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.1.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==2.8.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-tensorflow==2.1.0.1.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn==0.23.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib==3.2.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'boto3==1.14.60'])

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
import tensorflow as tf
import boto3

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers.configuration_distilbert import DistilBertConfig
from tensorflow.keras.models import load_model


CLASSES = [1, 2, 3, 4, 5]


def select_data_and_label_from_record(record):
    x = {
        'input_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'segment_ids': record['segment_ids']
    }

    y = record['label_ids']

    return (x, y)


def file_based_input_dataset_builder(channel,
                                     input_filenames,
                                     is_training,
                                     drop_remainder,
                                     batch_size,
                                     epochs,
                                     steps_per_epoch,
                                     max_seq_length):

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    print('***** Using input_filenames {}'.format(input_filenames))
    dataset = tf.data.TFRecordDataset(input_filenames)

    dataset = dataset.repeat(epochs * steps_per_epoch * 100)

    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        record = tf.io.parse_single_example(record, name_to_features)
        # TODO:  wip/bert/bert_attention_head_view/train.py
        # Convert input_ids into input_tokens with DistilBert vocabulary 
        #  if hook.get_collections()['all'].save_config.should_save_step(modes.EVAL, hook.mode_steps[modes.EVAL]):
        #    hook._write_raw_tensor_simple("input_tokens", input_tokens)
        return record
    
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
          lambda record: _decode_record(record, name_to_features),
          batch_size=batch_size,
          drop_remainder=drop_remainder,
          num_parallel_calls=tf.data.experimental.AUTOTUNE))

    dataset = dataset.shuffle(buffer_size=1000,
                              reshuffle_each_iteration=True)

    row_count = 0
    print('**************** {} *****************'.format(channel))
    for row in dataset.as_numpy_iterator():
        print(row)
        if row_count == 5:
            break
        row_count = row_count + 1

    return dataset

def upload_model_to_s3(file, bucket):
    s3 = boto3.client('s3')
    with open(file, "rb") as f:
    s3.upload_fileobj(f, bucket, file)
    print('Model file {} uploaded to {}'.format(file, bucket))


if __name__ == '__main__':

    train_data='s3://sagemaker-us-west-2-231218423789/training-pipeline-2020-09-05-16-19-31/processing/output/bert-train'
    test_data='s3://sagemaker-us-west-2-231218423789/training-pipeline-2020-09-05-16-19-31/processing/output/bert-test'
    validation_data='s3://sagemaker-us-west-2-231218423789/training-pipeline-2020-09-05-16-19-31/processing/output/bert-validation'
    model_dir='opt/ml/model'
    output_dir='s3://sagemaker-us-west-2-231218423789/dlc/output'
    use_xla=False
    use_amp=False
    max_seq_length=64
    train_batch_size=64
    validation_batch_size=64
    test_batch_size=64
    epochs=1
    learning_rate=0.00003
    epsilon=0.00000001
    train_steps_per_epoch=50
    validation_steps=10
    test_steps=10
    freeze_bert_layer=True
    run_validation=False
    run_test=False
    run_sample_predictions=False
 
    # Model Output 
    transformer_fine_tuned_model_path = os.path.join(model_dir, 'transformers/fine-tuned/')
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)

    # SavedModel Output
    tensorflow_saved_model_path = os.path.join(model_dir, 'tensorflow/saved_model/0')
    os.makedirs(tensorflow_saved_model_path, exist_ok=True)
    
    distributed_strategy = tf.distribute.MirroredStrategy()
    
    with distributed_strategy.scope():
        tf.config.optimizer.set_jit(use_xla)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": use_amp})

        train_data_filenames = glob(os.path.join(train_data, '*.tfrecord'))
        print('train_data_filenames {}'.format(train_data_filenames))
        train_dataset = file_based_input_dataset_builder(
            channel='train',
            input_filenames=train_data_filenames,
            is_training=True,
            drop_remainder=False,
            batch_size=train_batch_size,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            max_seq_length=max_seq_length).map(select_data_and_label_from_record)

        tokenizer = None
        config = None
        model = None

        # This is required when launching many instances at once...  the urllib request seems to get denied periodically
        successful_download = False
        retries = 0
        while (retries < 5 and not successful_download):
            try:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
                                                          num_labels=len(CLASSES))
                model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                              config=config)
                successful_download = True
                print('Sucessfully downloaded after {} retries.'.format(retries))
            except:
                retries = retries + 1
                random_sleep = random.randint(1, 30)
                print('Retry #{}.  Sleeping for {} seconds'.format(retries, random_sleep))
                time.sleep(random_sleep)

        callbacks = []

        initial_epoch_number = 0 

        if not tokenizer or not model or not config:
            print('Not properly initialized...')

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        print('** use_amp {}'.format(use_amp))        
        if use_amp:
            # loss scaling is currently required when using mixed precision
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')

        print('*** OPTIMIZER {} ***'.format(optimizer))
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        print('Compiled model {}'.format(model))          
        model.layers[0].trainable = not freeze_bert_layer
        print(model.summary())

        if run_validation:
            validation_data_filenames = glob(os.path.join(validation_data, '*.tfrecord'))
            print('validation_data_filenames {}'.format(validation_data_filenames))
            validation_dataset = file_based_input_dataset_builder(
                channel='validation',
                input_filenames=validation_data_filenames,
                is_training=False,
                drop_remainder=False,
                batch_size=validation_batch_size,
                epochs=epochs,
                steps_per_epoch=validation_steps,
                max_seq_length=max_seq_length).map(select_data_and_label_from_record)
            
            print('Starting Training and Validation...')
            validation_dataset = validation_dataset.take(validation_steps)
            train_and_validation_history = model.fit(train_dataset,
                                                     shuffle=True,
                                                     epochs=epochs,
                                                     initial_epoch=initial_epoch_number,
                                                     steps_per_epoch=train_steps_per_epoch,
                                                     validation_data=validation_dataset,
                                                     validation_steps=validation_steps,
                                                     callbacks=callbacks)                                
            print(train_and_validation_history)
        else: # Not running validation
            print('Starting Training (Without Validation)...')
            train_history = model.fit(train_dataset,
                                      shuffle=True,
                                      epochs=epochs,
                                      initial_epoch=initial_epoch_number,
                                      steps_per_epoch=train_steps_per_epoch,
                                      callbacks=callbacks)                
            print(train_history)

        if run_test:
            test_data_filenames = glob(os.path.join(test_data, '*.tfrecord'))
            print('test_data_filenames {}'.format(test_data_filenames))
            test_dataset = file_based_input_dataset_builder(
                channel='test',
                input_filenames=test_data_filenames,
                is_training=False,
                drop_remainder=False,
                batch_size=test_batch_size,
                epochs=epochs,
                steps_per_epoch=test_steps,
                max_seq_length=max_seq_length).map(select_data_and_label_from_record)

            print('Starting test...')
            test_history = model.evaluate(test_dataset,
                                          steps=test_steps,
                                          callbacks=callbacks)
                                 
            print('Test history {}'.format(test_history))
            
        # Save the Fine-Yuned Transformers Model as a New "Pre-Trained" Model
        print('transformer_fine_tuned_model_path {}'.format(transformer_fine_tuned_model_path))   
        model.save_pretrained(transformer_fine_tuned_model_path)
        upload_model_to_s3(transformer_fine_tuned_model_path, output_dir)

        # Save the TensorFlow SavedModel for Serving Predictions
        print('tensorflow_saved_model_path {}'.format(tensorflow_saved_model_path))   
        model.save(tensorflow_saved_model_path, save_format='tf')
        upload_model_to_s3(tensorflow_saved_model_path, output_dir)

                
        # Copy inference.py and requirements.txt to the code/ directory
        #   Note: This is required for the SageMaker Endpoint to pick them up.
        #         This appears to be hard-coded and must be called code/
        inference_path = os.path.join(local_model_dir, 'code/')
        print('Copying inference source files to {}'.format(inference_path))
        os.makedirs(inference_path, exist_ok=True)               
        os.system('cp inference.py {}'.format(inference_path))
        print(glob(inference_path))        
#        os.system('cp requirements.txt {}/code'.format(inference_path))
        
    if run_sample_predictions:
        loaded_model = TFDistilBertForSequenceClassification.from_pretrained(transformer_fine_tuned_model_path,
                                                                       id2label={
                                                                        0: 1,
                                                                        1: 2,
                                                                        2: 3,
                                                                        3: 4,
                                                                        4: 5
                                                                       },
                                                                       label2id={
                                                                        1: 0,
                                                                        2: 1,
                                                                        3: 2,
                                                                        4: 3,
                                                                        5: 4
                                                                       })

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        inference_pipeline = TextClassificationPipeline(model=loaded_model, 
                                                        tokenizer=tokenizer,
                                                        framework='tf',
                                                        device=-1)  

        print("""I loved it!  I will recommend this to everyone.""", inference_pipeline("""I loved it!  I will recommend this to everyone."""))
        print("""It's OK.""", inference_pipeline("""It's OK."""))
        print("""Really bad.  I hope they don't make this anymore.""", inference_pipeline("""Really bad.  I hope they don't make this anymore."""))

        import csv

        df_test_reviews = pd.read_csv('./test_data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', 
                                        delimiter='\t', 
                                        quoting=csv.QUOTE_NONE,
                                        compression='gzip')[['review_body', 'star_rating']]

        df_test_reviews = df_test_reviews.sample(n=100)
        df_test_reviews.shape
        df_test_reviews.head()
        
        import pandas as pd

        def predict(review_body):
            prediction_map = inference_pipeline(review_body)
            return prediction_map[0]['label']

        y_test = df_test_reviews['review_body'].map(predict)
        y_test
        
        y_actual = df_test_reviews['star_rating']
        y_actual

        from sklearn.metrics import classification_report
        print(classification_report(y_true=y_test, y_pred=y_actual))
        
        from sklearn.metrics import accuracy_score
        print('Accuracy: ', accuracy_score(y_true=y_test, y_pred=y_actual))
        
        import matplotlib.pyplot as plt
        import pandas as pd

        def plot_conf_mat(cm, classes, title, cmap = plt.cm.Greens):
            print(cm)
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="black" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                
        import itertools
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        #%matplotlib inline
        #%config InlineBackend.figure_format='retina'

        cm = confusion_matrix(y_true=y_test, y_pred=y_actual)

        plt.figure()
        fig, ax = plt.subplots(figsize=(10,5))
        plot_conf_mat(cm, 
                      classes=['1', '2', '3', '4', '5'], 
                      title='Confusion Matrix')

        # Save the confusion matrix        
        plt.show()
        
        # Model Output 
        metrics_path = os.path.join(local_model_dir, 'metrics/')
        os.makedirs(metrics_path, exist_ok=True)
        plt.savefig('{}/confusion_matrix.png'.format(metrics_path))