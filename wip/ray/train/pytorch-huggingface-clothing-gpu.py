#""" Finetuning a 🤗 Transformers model for sequence classification."""
import argparse
import logging
import math
import os
import random
import torch

import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5001')

from typing import Dict, Any

import datasets
import ray
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from ray.train import Trainer
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help=(
            "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated, sequences shorter "
            "will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic "
        "padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from "
        "huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 "
        "Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50,
        help="Total number of training steps to perform. If provided, "
        "overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a "
        "backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    # Ray arguments.
    parser.add_argument(
        "--start_local", action="store_true", help="Starts Ray on local machine."
    )
    parser.add_argument(
        "--address", type=str, default=None, help="Ray address to connect to."
    )
    parser.add_argument(
        "--num_workers", type=int, default=3, help="Number of workers to use."
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="If training should be done on GPUs."
    )

    args = parser.parse_args()

    # Sanity checks
    if (
#        args.task_name is None
        args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def train_func(config: Dict[str, Any]):
    args = config["args"]
    # Initialize the accelerator. We will let the accelerator handle device
    # placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on
    # the screen. accelerator.is_local_main_process is only True for one
    # process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and
    # evaluation files (see below) or specify a GLUE benchmark task (the
    # dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called
    # 'label' and as pair of sentences the sentences in columns called
    # 'sentence1' and 'sentence2' if such column exists or the first two
    # columns not named label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does
    # single sentence classification on this single column. You can easily
    # tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only
    # one local process can concurrently download the dataset.
#    if args.task_name is not None:
#        # Downloading and loading a dataset from the hub.
#        raw_datasets = load_dataset("glue", args.task_name)
#    else:
        # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (
        args.train_file if args.train_file is not None else args.valid_file
    ).split(".")[-1]

    raw_datasets = load_dataset(extension, data_files=data_files)

    label_list = raw_datasets["train"].unique("sentiment")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that
    # only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, 
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    # Preprocessing the datasets
    sentence1_key, sentence2_key = "review_body", None

    # Some models have set the order of the labels to use,
    # so let's make sure we do use it.
    label_to_id = None
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "sentiment" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [
                    label_to_id[l] for l in examples["sentiment"]  # noqa:E741
                ]
            else:
                # In all cases, rename the column to labels because the model
                # will expect that.
                result["labels"] = examples["sentiment"]

        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data
        # collator that will just convert everything to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for
        # us (by padding to the maximum length of the samples passed). When
        # using mixed precision, we add `pad_to_multiple_of=8` to pad all
        # tensors to multiple of 8s, which will enable the use of Tensor
        # Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

#    model, optimizer, train_dataloader = accelerator.prepare(
#        model, optimizer, train_dataloader
#    )
    # Note -> the training dataloader needs to be prepared before we grab
    # his length below (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
#    if args.task_name is not None:
#        metric = load_metric("glue", args.task_name)
#    else:
    metric = load_metric("accuracy")

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device ="
        f" {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) "
        f"= {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = (
                outputs.logits.argmax(dim=-1)
#                if not is_regression
#                else outputs.logits.squeeze()
            )
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        mlflow.log_param("use_gpu", args.use_gpu)
        mlflow.log_param("batch_size", args.per_device_train_batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("max_length", args.max_length)

        eval_metric = metric.compute()
        mlflow.log_metric("accuracy", eval_metric['accuracy'])

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


def main():
    args = parse_args()
    config = {"args": args}
    args.use_gpu = True
    if args.start_local or args.address or args.num_workers > 1 or args.use_gpu:
        if args.start_local:
            # Start a local Ray runtime.
            ray.init(num_cpus=args.num_workers)
        else:
            # Connect to a Ray cluster for distributed training.
            ray.init(address=args.address)
        trainer = Trainer("torch", num_workers=args.num_workers, use_gpu=args.use_gpu,
                          resources_per_worker={'CPU': 4, 'GPU': 1})
        trainer.start()
        trainer.run(train_func, config)
    else:
        # Run training locally.
        train_func(config)


if __name__ == "__main__":
    main()
