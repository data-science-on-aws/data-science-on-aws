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
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==2.8.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-tensorflow==2.1.0.1.0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'smdebug==0.7.2'])
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers.configuration_distilbert import DistilBertConfig

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
                                     pipe_mode,
                                     is_training,
                                     drop_remainder,
                                     batch_size,
                                     epochs,
                                     steps_per_epoch,
                                     max_seq_length):

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if pipe_mode:
        print('***** Using pipe_mode with channel {}'.format(channel))
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel,
                                  record_format='TFRecord')
    else:
        print('***** Using input_filenames {}'.format(input_filenames))
        dataset = tf.data.TFRecordDataset(input_filenames)

    dataset = dataset.repeat(epochs * steps_per_epoch)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
#      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        return tf.io.parse_single_example(record, name_to_features)
    
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
          lambda record: _decode_record(record, name_to_features),
          batch_size=batch_size,
          drop_remainder=drop_remainder,
          num_parallel_calls=tf.data.experimental.AUTOTUNE))

    dataset.cache()

    if is_training:
        dataset = dataset.shuffle(seed=42,
                                  buffer_size=1000,
                                  reshuffle_each_iteration=True)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test-data',
                        type=str,
                        default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--model-dir', 
                        type=str, 
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-data-dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--hosts', 
                        type=list, 
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', 
                        type=str, 
                        default=os.environ['SM_CURRENT_HOST'])    
    parser.add_argument('--num-gpus', 
                        type=int, 
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--use-xla',
                        type=bool,
                        default=False)
    parser.add_argument('--use-amp',
                        type=bool,
                        default=False)
    parser.add_argument('--max-seq-length',
                        type=int,
                        default=128)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=128)
    parser.add_argument('--validation-batch-size',
                        type=int,
                        default=256)
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=256)
    parser.add_argument('--epochs',
                        type=int,
                        default=2)
    parser.add_argument('--train-steps-per-epoch',
                        type=int,
                        default=100)
    parser.add_argument('--validation-steps',
                        type=int,
                        default=100)
    parser.add_argument('--test-steps',
                        type=int,
                        default=100)
    parser.add_argument('--freeze-bert-layer',
                        type=bool,
                        default=False)
    parser.add_argument('--enable-sagemaker-debugger',
                        type=bool,
                        default=False)
    parser.add_argument('--run-validation',
                        type=bool,
                        default=False)    
    parser.add_argument('--run-test',
                        type=bool,
                        default=False)    
    parser.add_argument('--run-sample-predictions',
                        type=bool,
                        default=False)    
#     parser.add_argument('--disable-eager-execution',
#                         type=bool,
#                         default=False) 
    
    args, _ = parser.parse_known_args()
    print("Args:") 
    print(args)
    
    env_var = os.environ 
    print("Environment Variables:") 
    pprint.pprint(dict(env_var), width = 1) 

    train_data = args.train_data
    validation_data = args.validation_data
    test_data = args.test_data
    model_dir = args.model_dir
    output_data_dir = args.output_data_dir
    hosts = args.hosts
    current_host = args.current_host
    num_gpus = args.num_gpus
    use_xla = args.use_xla
    use_amp = args.use_amp
    max_seq_length = args.max_seq_length
    train_batch_size = args.train_batch_size
    validation_batch_size = args.validation_batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    train_steps_per_epoch = args.train_steps_per_epoch
    validation_steps = args.validation_steps
    test_steps = args.test_steps
    freeze_bert_layer = args.freeze_bert_layer
    enable_sagemaker_debugger = args.enable_sagemaker_debugger
    run_validation = args.run_validation
    run_test = args.run_test
    run_sample_predictions = args.run_sample_predictions
#    disable_eager_execution = args.disable_eager_execution    

    # Determine if PipeMode is enabled 
    pipe_mode_str = os.environ.get('SM_INPUT_DATA_CONFIG', '')
    pipe_mode = (pipe_mode_str.find('Pipe') >= 0)
    print('Using pipe_mode: {}'.format(pipe_mode))
 
    # Model Output 
    transformer_pretrained_model_path = os.path.join(model_dir, 'transformer/pretrained')
    os.makedirs(transformer_pretrained_model_path, exist_ok=True)

    # SavedModel Output
    tensorflow_saved_model_path = os.path.join(model_dir, 'saved_model/0')
    os.makedirs(tensorflow_saved_model_path, exist_ok=True)

    # Tensorboard Logs 
    tensorboard_logs_path = os.path.join(output_data_dir, 'tensorboard') 
    os.makedirs(tensorboard_logs_path, exist_ok=True)

#     print('disable_eager_execution {}'.format(disable_eager_execution))
#     if disable_eager_execution: 
#         tf.compat.v1.disable_eager_execution()        
        
#    distributed_strategy = tf.distribute.MirroredStrategy()
    distributed_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with distributed_strategy.scope():
        tf.config.optimizer.set_jit(use_xla)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": use_amp})

        train_data_filenames = glob(os.path.join(train_data, '*.tfrecord'))
        print('train_data_filenames {}'.format(train_data_filenames))
        train_dataset = file_based_input_dataset_builder(
            channel='train',
            input_filenames=train_data_filenames,
            pipe_mode=pipe_mode,
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

        if not tokenizer or not model or not config:
            print('Not properly initialized...')

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
        if use_amp:
            # loss scaling is currently required when using mixed precision
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')

        callbacks = []
        if enable_sagemaker_debugger:
            import smdebug.tensorflow as smd
            callback = smd.KerasHook(out_dir=output_data_dir,
                                     export_tensorboard=True,        
                                     tensorboard_dir=tensorboard_logs_path,
                                     save_config=smd.SaveConfig(save_interval=100),
#                                     save_all=True,
                                     include_collections=['metrics', 
                                                          'losses', 
                                                          'sm_metrics'],
                                     include_workers='all')
            callbacks.append(callback)
            optimizer = callback.wrap_optimizer(optimizer)
        else:
            callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
            callbacks.append(callback)
            
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        print('Trained model {}'.format(model))

        if run_validation:
            validation_data_filenames = glob(os.path.join(validation_data, '*.tfrecord'))
            print('validation_data_filenames {}'.format(validation_data_filenames))
            validation_dataset = file_based_input_dataset_builder(
                channel='validation',
                input_filenames=validation_data_filenames,
                pipe_mode=pipe_mode,
                is_training=False,
                drop_remainder=False,
                batch_size=validation_batch_size,
                epochs=epochs,
                steps_per_epoch=validation_steps,
                max_seq_length=max_seq_length).map(select_data_and_label_from_record)

            train_and_validation_history = model.fit(train_dataset,
                                                     shuffle=True,
                                                     epochs=epochs,
                                                     steps_per_epoch=train_steps_per_epoch,
                                                     validation_data=validation_dataset,
                                                     validation_steps=validation_steps,
                                                     callbacks=callbacks)
            print(train_and_validation_history)
        else: # Not running validation
            train_history = model.fit(train_dataset,
                                      shuffle=True,
                                      epochs=epochs,
                                      steps_per_epoch=train_steps_per_epoch,
                                      callbacks=callbacks)
            print(train_history)

        if run_test:
            test_data_filenames = glob(os.path.join(test_data, '*.tfrecord'))
            print('test_data_filenames {}'.format(test_data_filenames))
            test_dataset = file_based_input_dataset_builder(
                channel='test',
                input_filenames=test_data_filenames,
                pipe_mode=pipe_mode,
                is_training=False,
                drop_remainder=False,
                batch_size=test_batch_size,
                epochs=epochs,
                steps_per_epoch=test_steps,
                max_seq_length=max_seq_length).map(select_data_and_label_from_record)

            test_history = model.evaluate(test_dataset,
                                          steps=test_steps,
                                          callbacks=callbacks)
            print(test_history)

            
        # Save the fine-tuned Transformers Model
        model.save_pretrained(transformer_pretrained_model_path)
        # Save the TensorFlow SavedModel
        model.save(tensorflow_saved_model_path, save_format='tf')

    if run_sample_predictions:
        loaded_model = TFDistilBertForSequenceClassification.from_pretrained(transformer_pretrained_model_path,
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

        if num_gpus >= 1:
            inference_device = 0 # GPU 0
        else:
            inference_device = -1 # CPU
        print('inference_device {}'.format(inference_device))

        inference_pipeline = TextClassificationPipeline(model=loaded_model, 
                                                        tokenizer=tokenizer,
                                                        framework='tf',
                                                        device=inference_device)  

        print("""I loved it!  I will recommend this to everyone.""", inference_pipeline("""I loved it!  I will recommend this to everyone."""))
        print("""It's OK.""", inference_pipeline("""It's OK."""))
        print("""Really bad.  I hope they don't make this anymore.""", inference_pipeline("""Really bad.  I hope they don't make this anymore."""))
