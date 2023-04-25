import argparse
from functools import partial
from pathlib import Path
import os
import datetime

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import LoggerType
from datasets import concatenate_datasets, load_dataset
from nltk.tokenize import sent_tokenize
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="google/flan-t5-small",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/opt/ml/input/data/train",
        # required=True,
        help="Path to the dataset.",
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps.")
    parser.add_argument(
        "--subsample", type=int, default=25, help="percentage of training data to use."
    )
    parser.add_argument(
        "--model_dir", type=str, default="/opt/ml/model", help="Model dir."
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="/opt/ml/output/tensorboard",
        help="Tensorboard dir.",
    )
    parser.add_argument("--log_steps", type=int, default=10, help="Log interval steps.")

    args = parser.parse_args()

    return args


# def preprocess_function(
#     sample,
#     tokenizer,
#     max_source_length,
#     max_target_length,
#     padding="max_length",
# ):
#     # add prefix to the input for t5
#     inputs = ["summarize: " + item for item in sample["dialogue"]]

#     # tokenize inputs
#     model_inputs = tokenizer(
#         inputs, max_length=max_source_length, padding=padding, truncation=True
#     )

#     # Tokenize targets with the `text_target` keyword argument
#     labels = tokenizer(
#         text_target=sample["summary"],
#         max_length=max_target_length,
#         padding=padding,
#         truncation=True,
#     )

#     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
#     # padding in the loss.
#     if padding == "max_length":
#         labels["input_ids"] = [
#             [(l if l != tokenizer.pad_token_id else -100) for l in label]
#             for label in labels["input_ids"]
#         ]

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


def collate_fn(examples, tokenizer):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, labels, tokenizer):
    
    metric = evaluate.load("rouge")

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def list_files(startpath):
    """Helper function to list files in a directory"""
    print('Listing files for {}'.format(startpath))
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            

def main(args):

    model_name_or_path = args.pretrained_model_name_or_path
    dataset_path = Path(args.train_dataset_path)
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed = args.seed
    tb_log_dir = args.tensorboard_dir
    tb_log_interval = args.log_steps
    max_steps = args.max_steps

    accelerator = Accelerator(log_with=LoggerType.TENSORBOARD, project_dir=tb_log_dir)

    accelerator.init_trackers(".", init_kwargs={"tensorboard": {"flush_secs": 30}})

    config = {
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
    }

    if accelerator.is_main_process:
        # workaround for hparams not showing up in tensorboard if no metrics are logged
        # https://github.com/tensorflow/tensorboard/issues/2942
        tb_tracker = [
            tracker for tracker in accelerator.trackers if tracker.name == "tensorboard"
        ][0]

        # log hp_metric till the issue in TB is fixed
        tb_tracker.writer.add_hparams(config, {"hp_metric": 0}, run_name=".")
        tb_tracker.writer.flush()
        
    with accelerator.main_process_first():
        # configure evaluation metrics
        # this should run in main process first to download the punkt corpus
        nltk.download("punkt")

    set_seed(seed)

    # explore the input files
    local_data_processed_path = '/opt/ml/input/data'
    print('Listing all input data files...')
    list_files(local_data_processed_path)
    
    # # read the dataset
    # ds = datasets.Dataset.from_json((dataset_path / "dialogsum.train.jsonl").as_posix())
    
    # load the dataset
    print(f'loading dataset from: {local_data_processed_path}')
    processed_datasets = load_dataset(
        local_data_processed_path,
        data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
    ).with_format("torch")
    print(f'loaded datasets: {processed_datasets}')
        
    # # take a subsample of the data
    # ds = datasets.Dataset.from_pandas(
    #     ds.to_pandas().sample(
    #         frac=args.subsample / 100, random_state=seed, ignore_index=True
    #     )
    # )
        
#     # split into train and test
#     dataset = ds.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

#     tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
#         lambda x: tokenizer(x["dialogue"], truncation=True),
#         batched=True,
#         remove_columns=["dialogue", "summary", "fname", "topic"],
#     )
#     max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

#     tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
#         lambda x: tokenizer(x["summary"], truncation=True),
#         batched=True,
#         remove_columns=["dialogue", "summary", "fname", "topic"],
#     )
#     max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

#     preprocess = partial(
#         preprocess_function,
#         tokenizer=tokenizer,
#         max_source_length=max_source_length,
#         max_target_length=max_target_length,
#     )

#     # with accelerator.main_process_first():
#     processed_datasets = dataset.map(
#         preprocess,
#         batched=True,
#         num_proc=1,
#         load_from_cache_file=True,
#         remove_columns=["dialogue", "summary", "fname", "topic"],
#         desc="Running tokenizer on dataset",
#     )

    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]
    validation_dataset = processed_datasets["validation"]

    collate = partial(collate_fn, tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        collate_fn=collate,
        batch_size=batch_size,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset, 
        collate_fn=collate, 
        batch_size=batch_size * 8, 
        pin_memory=True
    )

    # create the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,  # size of the LoRA attention dimension
        lora_alpha=32,  # the gradients will be scaled by r / lora_alpha (similar to tuning the learning rate)
        lora_dropout=0.1,  # drop out rate for the LoRA attention
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # create the optimizer optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # create an lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # prepare model for training
    (
        model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        lr_scheduler,
    )

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    total_steps = 0
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        # TODO:  Use max_steps
        for step, batch in enumerate(tqdm(train_dataloader)):
            if step > max_steps:
                break
                
            # gradient accumulation
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss = total_loss / (step + 1)
            train_perplexity = torch.exp(train_loss)
            
            # log to tensorboard
            if step % tb_log_interval == 0:
                accelerator.log(
                    {
                        "training_loss": train_loss.item(),
                        "train_perplexity": train_perplexity.item(),
                    },
                    step=total_steps,
                )
            total_steps += 1

        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_perplexity = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_epoch_perplexity=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        eval_labels = []
        max_new_eval_tokens = 100
        for _, batch in enumerate(tqdm(validation_dataloader)):
            labels = batch.pop("labels")

            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch,
                    synced_gpus=is_ds_zero_3,
                    max_new_tokens=max_new_eval_tokens,
                )  # synced_gpus=True for DS-stage 3
                

            # pad outputs to max length
            outputs = torch.nn.functional.pad(
                outputs, (0, max_new_eval_tokens - outputs.shape[1]), "constant", tokenizer.pad_token_id
            )

            preds = accelerator.gather(outputs).detach().cpu().numpy()
            labels = accelerator.gather(labels).detach().cpu().numpy()

            eval_preds.extend(preds)
            eval_labels.extend(labels)

        if accelerator.is_main_process:
            eval_preds = np.stack(eval_preds)
            eval_labels = np.stack(eval_labels)
            metrics = compute_metrics(eval_preds, eval_labels, tokenizer)
            accelerator.print(metrics)
            accelerator.log(metrics, step=total_steps)

        accelerator.wait_for_everyone()

        peft_model_id = (
            f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
        )
        checkpoint_name = f"{args.model_dir}/{peft_model_id}/adapter_model.bin"

        if accelerator.is_main_process:
            model.save_pretrained(f"{args.model_dir}/{peft_model_id}")
            tokenizer.save_pretrained(f"{args.model_dir}/{peft_model_id}")
            
            try:
                #local_model_dir = args.model_dir
                # inference_path = os.path.join(local_model_dir, "code/")            
                # print("Copying inference source files to {}".format(inference_path))
                # os.makedirs(inference_path, exist_ok=True)
                # os.system("cp inference.py {}".format(inference_path))
                # os.system('cp requirements.txt {}'.format(inference_path))
                list_files(args.model_dir)
                os.system('cd inference && cp -R * {}'.format(args.model_dir))
            except:
                print('failed copy cd inference')

            try:
                # Copy test data for the evaluation step
                os.system("cp -R ./inference/* {}".format(args.model_dir))
                #print(f'Files in inference code path "{args.model_dir}"')
                list_files(args.model_dir)
            except:
                print('failed copy cp -R ./inference/*')

        accelerator.save(
            get_peft_model_state_dict(
                model, state_dict=accelerator.get_state_dict(model)
            ),
            checkpoint_name,
        )

        accelerator.wait_for_everyone()
        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
