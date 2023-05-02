
import argparse
from contextlib import contextmanager
import torch
import os
import json
from transformers import (
    MODEL_MAPPING,
    SchedulerType
)
# Distributed training helper methods.


def wait_for_everyone():
    #deepspeed.comm.barrier
    torch.distributed.barrier()


def is_main_process(rank):
    if rank == 0:
        return True
    else:
        return False

def _goes_first(is_main):
    if not is_main:
        wait_for_everyone()
    yield
    if is_main:
        wait_for_everyone()

@contextmanager
def main_process_first(rank):
    """
    Lets the main process go first inside a with block.
    The other processes will enter the with block after the main process exits.
    """
    yield from _goes_first(is_main_process(rank))

def is_local_main_process(local_rank):
    return local_rank == 0


# args parsing

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a FLAN T5 model on a Seq2Seq task")

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=100, help="A seed for reproducible training.")

    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument("--group_texts",default=False,help="Whether to group texts together when tokenizing")
  

    parser.add_argument("model_dir",type=str,default="/opt/ml/model")

    args,_ = parser.parse_known_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or json file."

    num_nodes = 0
    sm_config_path = '/opt/ml/input/config/resourceconfig.json'
    if os.path.exists(sm_config_path):
        with open(sm_config_path) as file:
            cluster_config = json.load(file)

        hosts = cluster_config['hosts']
        print("*****printing list of hosts **********")
        print(hosts)
        num_nodes = len(hosts)
        print("Total number of nodes in the training cluster - {}".format(num_nodes))
       
 
    args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
    args.local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))
    
    args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    args.world_size = num_nodes*args.local_size

    print("local rank : {}, global rank : {} , local size : {},  world size : {}".format(args.local_rank,args.rank, args.local_size, args.world_size))

    return args