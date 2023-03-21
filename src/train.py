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
import csv

import pandas as pd
import numpy as np

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1", "torchdata==0.5.1"])
import torch

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1", "datasets==2.9.0"])
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


# def load_checkpoint_model(checkpoint_path):
#     import glob
#     import os

#     glob_pattern = os.path.join(checkpoint_path, "*.h5")
#     print("glob pattern {}".format(glob_pattern))

#     list_of_checkpoint_files = glob.glob(glob_pattern)
#     print("List of checkpoint files {}".format(list_of_checkpoint_files))

#     latest_checkpoint_file = max(list_of_checkpoint_files)
#     print("Latest checkpoint file {}".format(latest_checkpoint_file))

#     initial_epoch_number_str = latest_checkpoint_file.rsplit("_", 1)[-1].split(".h5")[0]
#     initial_epoch_number = int(initial_epoch_number_str)

#     loaded_model = TFDistilBertForSequenceClassification.from_pretrained(latest_checkpoint_file, config=config)

#     print("loaded_model {}".format(loaded_model))
#     print("initial_epoch_number {}".format(initial_epoch_number))

#     return loaded_model, initial_epoch_number




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--test_steps", type=int, default=None)
    parser.add_argument("--enable_sagemaker_debugger", type=eval, default=False)
    parser.add_argument("--run_validation", type=eval, default=False)
    parser.add_argument("--run_test", type=eval, default=False)
    parser.add_argument("--run_sample_predictions", type=eval, default=False)
    parser.add_argument("--enable_tensorboard", type=eval, default=False)
    parser.add_argument("--enable_checkpointing", type=eval, default=False)
    parser.add_argument("--model_checkpoint", type=str, default=None)    
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])  # This is unused


    args, _ = parser.parse_known_args()
    print("Args:")
    print(args)

    env_var = os.environ
    print("Environment Variables:")
    pprint.pprint(dict(env_var), width=1)

    print("SM_TRAINING_ENV {}".format(env_var["SM_TRAINING_ENV"]))
    sm_training_env_json = json.loads(env_var["SM_TRAINING_ENV"])
    is_master = sm_training_env_json["is_master"]
    print("is_master {}".format(is_master))

    train_data = args.train_data
    print("train_data {}".format(train_data))
    validation_data = args.validation_data
    print("validation_data {}".format(validation_data))
    test_data = args.test_data
    print("test_data {}".format(test_data))
    local_model_dir = os.environ["SM_MODEL_DIR"]
    output_dir = args.output_dir
    print("output_dir {}".format(output_dir))
    hosts = args.hosts
    print("hosts {}".format(hosts))
    current_host = args.current_host
    print("current_host {}".format(current_host))
    num_gpus = args.num_gpus
    print("num_gpus {}".format(num_gpus))
    job_name = os.environ["SAGEMAKER_JOB_NAME"]
    print("job_name {}".format(job_name))
    train_batch_size = args.train_batch_size
    print("train_batch_size {}".format(train_batch_size))
    validation_batch_size = args.validation_batch_size
    print("validation_batch_size {}".format(validation_batch_size))
    test_batch_size = args.test_batch_size
    print("test_batch_size {}".format(test_batch_size))
    epochs = args.epochs
    print("epochs {}".format(epochs))
    learning_rate = args.learning_rate
    print("learning_rate {}".format(learning_rate))
    weight_decay = args.weight_decay
    print("weight_decay {}".format(weight_decay))
    train_steps_per_epoch = args.train_steps_per_epoch
    print("train_steps_per_epoch {}".format(train_steps_per_epoch))
    validation_steps = args.validation_steps
    print("validation_steps {}".format(validation_steps))
    test_steps = args.test_steps
    print("test_steps {}".format(test_steps))
    enable_sagemaker_debugger = args.enable_sagemaker_debugger
    print("enable_sagemaker_debugger {}".format(enable_sagemaker_debugger))
    run_validation = args.run_validation
    print("run_validation {}".format(run_validation))
    run_test = args.run_test
    print("run_test {}".format(run_test))
    run_sample_predictions = args.run_sample_predictions
    print("run_sample_predictions {}".format(run_sample_predictions))
    enable_tensorboard = args.enable_tensorboard
    print("enable_tensorboard {}".format(enable_tensorboard))
    enable_checkpointing = args.enable_checkpointing
    print("enable_checkpointing {}".format(enable_checkpointing))
    model_checkpoint = args.model_checkpoint
    print("model_checkpoint {}".format(model_checkpoint))

    checkpoint_base_path = args.checkpoint_base_path
    print("checkpoint_base_path {}".format(checkpoint_base_path))

    if is_master:
        checkpoint_path = checkpoint_base_path
    else:
        checkpoint_path = "/tmp/checkpoints"
    print("checkpoint_path {}".format(checkpoint_path))

    # Determine if PipeMode is enabled
    pipe_mode_str = os.environ.get("SM_INPUT_DATA_CONFIG", "")
    pipe_mode = pipe_mode_str.find("Pipe") >= 0
    print("Using pipe_mode: {}".format(pipe_mode))


    # Model Output
    transformer_fine_tuned_model_path = os.path.join(local_model_dir) #, "transformers/fine-tuned/")
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)

    # Tensorboard Logs
    tensorboard_logs_path = os.path.join(local_model_dir, "tensorboard/")
    os.makedirs(tensorboard_logs_path, exist_ok=True)

    tokenizer = None
    model = None

    # This is required when launching many instances at once...  the urllib request seems to get denied periodically
    successful_download = False
    retries = 0
    while retries < 5 and not successful_download:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

            successful_download = True
            print("Sucessfully downloaded after {} retries.".format(retries))
        except:
            retries = retries + 1
            random_sleep = random.randint(1, 30)
            print("Retry #{}.  Sleeping for {} seconds".format(retries, random_sleep))
            time.sleep(random_sleep)

    callbacks = []

#     initial_epoch_number = 0

#     if enable_checkpointing:
#         print("***** Checkpoint enabled *****")

#         os.makedirs(checkpoint_path, exist_ok=True)
#         if os.listdir(checkpoint_path):
#             print("***** Found checkpoint *****")
#             print(checkpoint_path)
#             model, initial_epoch_number = load_checkpoint_model(checkpoint_path)
#             print("***** Using checkpoint model {} *****".format(model))

#         checkpoint_callback = ModelCheckpoint(
#             filepath=os.path.join(checkpoint_path, "tf_model_{epoch:05d}.h5"),
#             save_weights_only=False,
#             verbose=1,
#             monitor="val_accuracy",
#         )
#         print("*** CHECKPOINT CALLBACK {} ***".format(checkpoint_callback))
#         callbacks.append(checkpoint_callback)

    if not tokenizer or not model:
        print("Not properly initialized...")

        
    print("enable_sagemaker_debugger {}".format(enable_sagemaker_debugger))
    if enable_sagemaker_debugger:
        print("*** DEBUGGING ***")
        import smdebug.pytorch as smd

        # This assumes that we specified debugger_hook_config
        debugger_callback = smd.KerasHook.create_from_json_file()
        print("*** DEBUGGER CALLBACK {} ***".format(debugger_callback))
        callbacks.append(debugger_callback)
        optimizer = debugger_callback.wrap_optimizer(optimizer)

    if enable_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
        print("*** TENSORBOARD CALLBACK {} ***".format(tensorboard_callback))
        callbacks.append(tensorboard_callback)


    print("Loading datasets...")

    print("train_data {}".format(train_data))
    print("validation_data {}".format(validation_data))
    print("test_data {}".format(test_data))

    from datasets import Dataset

    lm_dataset_train = Dataset.from_parquet('{}/*.parquet'.format(train_data))
                
    model_name = model_checkpoint.split("/")[-1]
    
    print("lm_dataset_train {}".format(lm_dataset_train))

    print("Setting up Training...")

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        f"{model_name}-finetuned-amazon-customer-reviews",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=train_steps_per_epoch,
        num_train_epochs=epochs,
        no_cuda=not torch.cuda.is_available()
    )
    
    from transformers import Trainer
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset_train,
    )

    trainer.train()


    # Save the Fine-tuned Transformers Model as a New "Pre-Trained" Model
    print("transformer_fine_tuned_model_path {}".format(transformer_fine_tuned_model_path))
    model.save_pretrained(transformer_fine_tuned_model_path)
    tokenizer.save_pretrained(transformer_fine_tuned_model_path)

    # Copy inference.py and requirements.txt to the code/ directory
    #   Note: This is required for the SageMaker Endpoint to pick them up.
    #         This appears to be hard-coded and must be called code/
    inference_path = os.path.join(local_model_dir, "code/")
    print("Copying inference source files to {}".format(inference_path))
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
    print(glob(inference_path))
    os.system('cp requirements.txt {}'.format(inference_path))

    # Copy test data for the evaluation step
    os.system("cp -R ./test_data/ {}".format(local_model_dir))

    
    if run_sample_predictions:
        def predict(text):
            result_length = 100
            inputs = tokenizer(text, return_tensors='pt')

            return tokenizer.decode(model.generate(inputs["input_ids"],
                                   max_length=result_length, 
                                   do_sample=True, 
                                   top_k=50, 
                                   top_p=0.9
                                  )[0])

        print(
            """Write a review for Norton Antivirus.""",
            predict("""Write a review for Norton Antivirus.""",),
        )
        
        print(
            """Write a review for TurboTax.""",
            predict("""Write a review for TurboTax.""",),
        )
        
