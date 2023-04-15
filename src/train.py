import argparse
import os
import json
import pprint

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig
from datasets import load_dataset

def list_files(startpath):
    """Helper function to list files in a directory"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def parse_args():
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

    return args


if __name__ == "__main__":
    
    # parse arguments
    args = parse_args()
    
    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    
    # explore the input files
    local_data_processed_path = '/opt/ml/input/data'
    print('Listing all input data files...')
    list_files(local_data_processed_path)
    
    # load the dataset
    print(f'loading dataset from: {local_data_processed_path}')
    tokenized_dataset = load_dataset(
        local_data_processed_path,
        data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
    ).with_format("torch")
    print(f'loaded dataset: {tokenized_dataset}')
    
    # train the model
    # (remove this filtering if you want to train for a longer period)
    sample_tokenized_dataset = tokenized_dataset.filter(lambda example, indice: indice % 100 == 0, with_indices=True)

    output_dir = args.checkpoint_base_path
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.validation_batch_size,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sample_tokenized_dataset['train'],
        eval_dataset=sample_tokenized_dataset['validation']
    )
    trainer.train()
    
    # save the model
    transformer_fine_tuned_model_path = os.environ["SM_MODEL_DIR"]
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)
    print(f"Saving the final model to: transformer_fine_tuned_model_path={transformer_fine_tuned_model_path}")
    model.save_pretrained(transformer_fine_tuned_model_path)
    tokenizer.save_pretrained(transformer_fine_tuned_model_path)
    

############################################################################################################
# if __name__ == "__main__":
    
#     args = parse_args()
    
#     checkpoint_base_path = args.checkpoint_base_path
#     print("checkpoint_base_path {}".format(checkpoint_base_path))

#     if is_master:
#         checkpoint_path = checkpoint_base_path
#     else:
#         checkpoint_path = "/tmp/checkpoints"
#     print("checkpoint_path {}".format(checkpoint_path))


#     # Determine if PipeMode is enabled
#     pipe_mode_str = os.environ.get("SM_INPUT_DATA_CONFIG", "")
#     pipe_mode = pipe_mode_str.find("Pipe") >= 0
#     print("Using pipe_mode: {}".format(pipe_mode))


#     # Model Output
#     transformer_fine_tuned_model_path = os.path.join(local_model_dir) #, "transformers/fine-tuned/")
#     os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)

#     tokenizer = None
#     model = None

#     # This is required when launching many instances at once...  the urllib request seems to get denied periodically
#     successful_download = False
#     retries = 0
#     while retries < 5 and not successful_download:
#         try:
#             from transformers import AutoTokenizer

#             tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
#             model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

#             successful_download = True
#             print("Sucessfully downloaded after {} retries.".format(retries))
#         except:
#             retries = retries + 1
#             random_sleep = random.randint(1, 30)
#             print("Retry #{}.  Sleeping for {} seconds".format(retries, random_sleep))
#             time.sleep(random_sleep)

#     callbacks = []

#     if not tokenizer or not model:
#         print("Not properly initialized...")

#     print("enable_sagemaker_debugger {}".format(enable_sagemaker_debugger))
#     if enable_sagemaker_debugger:
#         print("*** DEBUGGING ***")
#         import smdebug.pytorch as smd

#         # This assumes that we specified debugger_hook_config
#         debugger_callback = smd.KerasHook.create_from_json_file()
#         print("*** DEBUGGER CALLBACK {} ***".format(debugger_callback))
#         callbacks.append(debugger_callback)
#         optimizer = debugger_callback.wrap_optimizer(optimizer)

#     if enable_tensorboard:
#         tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
#         print("*** TENSORBOARD CALLBACK {} ***".format(tensorboard_callback))
#         callbacks.append(tensorboard_callback)


#     print("Loading datasets...")

#     print("train_data {}".format(train_data))
#     print("validation_data {}".format(validation_data))
#     print("test_data {}".format(test_data))

#     from datasets import Dataset

#     lm_dataset_train = Dataset.from_parquet('{}/*.parquet'.format(train_data))
                
#     model_name = model_checkpoint.split("/")[-1]
    
#     print("lm_dataset_train {}".format(lm_dataset_train))

#     print("Setting up Training...")

#     from transformers import TrainingArguments

#     training_args = TrainingArguments(
#         f"{model_name}-finetuned-amazon-customer-reviews",
#         learning_rate=learning_rate,
#         weight_decay=weight_decay,
#         max_steps=train_steps_per_epoch,
#         num_train_epochs=epochs,
#         no_cuda=not torch.cuda.is_available()
#     )
    
#     from transformers import Trainer
        
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=lm_dataset_train,
#     )

#     trainer.train()


#     # Save the Fine-tuned Transformers Model as a New "Pre-Trained" Model
#     print("transformer_fine_tuned_model_path {}".format(transformer_fine_tuned_model_path))
#     model.save_pretrained(transformer_fine_tuned_model_path)
#     tokenizer.save_pretrained(transformer_fine_tuned_model_path)

#     # Copy inference.py and requirements.txt to the code/ directory
#     #   Note: This is required for the SageMaker Endpoint to pick them up.
#     #         This appears to be hard-coded and must be called code/
#     inference_path = os.path.join(local_model_dir, "code/")
#     print("Copying inference source files to {}".format(inference_path))
#     os.makedirs(inference_path, exist_ok=True)
#     os.system("cp inference.py {}".format(inference_path))
#     print(glob(inference_path))
#     os.system('cp requirements.txt {}'.format(inference_path))

#     # Copy test data for the evaluation step
#     os.system("cp -R ./test_data/ {}".format(local_model_dir))

    
#     if run_sample_predictions:
#         def predict(text):
#             result_length = 100
#             inputs = tokenizer(text, return_tensors='pt')

#             return tokenizer.decode(model.generate(inputs["input_ids"],
#                                    max_length=result_length, 
#                                    do_sample=True, 
#                                    top_k=50, 
#                                    top_p=0.9
#                                   )[0])

#         print(
#             """Write a review for Norton Antivirus.""",
#             predict("""Write a review for Norton Antivirus.""",),
#         )
        
#         print(
#             """Write a review for TurboTax.""",
#             predict("""Write a review for TurboTax.""",),
#         )
        
