import sys
import os
import subprocess
import json
import sys
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

# hyperparameters defined in estimator as passed to your training container
# and saved in /opt/ml/input/config/hyperparameters.json. SM reads these hyperparameters 
# and parse_args() is used to extract them
def parse_args():
    parser = ArgumentParser(
        description=("SageMaker DeepSpeed Launch helper utility that will spawn deepspeed training scripts")
    )
    parser.add_argument(
        "--training_script",
        type=str,
        help="Path to the training program/script to be run in parallel, can be either absolute or relative",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the accelerate config file",
    )

    # rest from the training program
    parsed, nargs = parser.parse_known_args()

    return parsed.training_script, parsed.config_file, nargs


def accelerate_launch(command):
    try:
        proc = subprocess.run(command, shell=True)
    except Exception as e:
        logger.info(e)
    
    sys.exit(proc.returncode)

def main():

    train_script, config_file, args = parse_args()
    command = f"accelerate launch --config_file {config_file} {train_script} {' '.join(args)}"
    print(f"command = {command}")
    # launch deepspeed training
    accelerate_launch(command)


if __name__ == "__main__":
    main()
