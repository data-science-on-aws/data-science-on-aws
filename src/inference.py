import json
import logging
from typing import Any
from typing import Dict
from typing import Union
import subprocess
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from sagemaker_inference import encoder
from transformers import TextGenerationPipeline
from transformers import pipeline
from transformers import set_seed


APPLICATION_X_TEXT = "application/x-text"
APPLICATION_JSON = "application/json"
STR_DECODE_CODE = "utf-8"

VERBOSE_EXTENSION = ";verbose"

TEXT_GENERATION = "text-generation"

GENERATED_TEXT = "generated_text"
GENERATED_TEXTS = "generated_texts"

# Possible model parameters
TEXT_INPUTS = "text_inputs"
MAX_LENGTH = "max_length"
NUM_RETURN_SEQUENCES = "num_return_sequences"
NUM_NEW_TOKENS = "num_new_tokens"
NUM_BEAMS = "num_beams"
TOP_P = "top_p"
EARLY_STOPPING = "early_stopping"
DO_SAMPLE = "do_sample"
NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
TOP_K = "top_k"
TEMPERATURE = "temperature"
SEED = "seed"

ALL_PARAM_NAMES = [
    TEXT_INPUTS,
    MAX_LENGTH,
    NUM_NEW_TOKENS,
    NUM_RETURN_SEQUENCES,
    NUM_BEAMS,
    TOP_P,
    EARLY_STOPPING,
    DO_SAMPLE,
    NO_REPEAT_NGRAM_SIZE,
    TOP_K,
    TEMPERATURE,
    SEED,
]


# Model parameter ranges
MAX_LENGTH_MIN = 1
NUM_RETURN_SEQUENCE_MIN = 1
NUM_BEAMS_MIN = 1
TOP_P_MIN = 0
TOP_P_MAX = 1
NO_REPEAT_NGRAM_SIZE_MIN = 1
TOP_K_MIN = 0
TEMPERATURE_MIN = 0




def model_fn(model_dir: str) -> list:
    """Create our inference task as a delegate to the model.

    This runs only once per one worker.

    Args:
        model_dir (str): directory where the model files are stored
    Returns:
        list: a huggingface tokenizer and model
    """
    
    print('walking model_dir: {}'.format(model_dir))

    import os
    for root, dirs, files in os.walk(model_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
            
    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    print(f'Loaded Local HuggingFace Tokenzier:\n{tokenizer}')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    print(f'Loaded Local HuggingFace Model:\n{model}')
    
    return [tokenizer, model]


def _validate_payload(payload: Dict[str, Any]) -> None:
    """Validate the parameters in the input loads.

    Checks if max_length, num_return_sequences, num_beams, top_p and temprature are in bounds.
    Checks if do_sample is boolean.
    Checks max_length, num_return_sequences, num_beams and seed are integers.

    Args:
        payload: a decoded input payload (dictionary of input parameter and values)
    """
    for param_name in payload:
        assert (
            param_name in ALL_PARAM_NAMES
        ), f"Input payload contains an invalid key {param_name}. Valid keys are {ALL_PARAM_NAMES}."

    assert TEXT_INPUTS in payload, f"Input payload must contain {TEXT_INPUTS} key."

    for param_name in [MAX_LENGTH, NUM_RETURN_SEQUENCES, NUM_BEAMS, SEED]:
        if param_name in payload:
            assert type(payload[param_name]) == int, f"{param_name} must be an integer, got {payload[param_name]}."

    if MAX_LENGTH in payload:
        assert (
            payload[MAX_LENGTH] >= MAX_LENGTH_MIN
        ), f"{MAX_LENGTH} must be at least {MAX_LENGTH_MIN}, got {payload[MAX_LENGTH]}."
    if NUM_RETURN_SEQUENCES in payload:
        assert payload[NUM_RETURN_SEQUENCES] >= NUM_RETURN_SEQUENCE_MIN, (
            f"{NUM_RETURN_SEQUENCES} must be at least {NUM_RETURN_SEQUENCE_MIN}, "
            f"got {payload[NUM_RETURN_SEQUENCES]}."
        )
    if NUM_BEAMS in payload:
        assert (
            payload[NUM_BEAMS] >= NUM_BEAMS_MIN
        ), f"{NUM_BEAMS} must be at least {NUM_BEAMS_MIN}, got {payload[NUM_BEAMS]}."
    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS in payload:
        assert payload[NUM_RETURN_SEQUENCES] <= payload[NUM_BEAMS], (
            f"{NUM_BEAMS} must be at least {NUM_RETURN_SEQUENCES}. Instead got "
            f"{NUM_BEAMS}={payload[NUM_BEAMS]} and {NUM_RETURN_SEQUENCES}="
            f"{payload[NUM_RETURN_SEQUENCES]}."
        )
    if TOP_P in payload:
        assert TOP_P_MIN <= payload[TOP_P] <= TOP_P_MAX, (
            f"{TOP_K} must be in range [{TOP_P_MIN},{TOP_P_MAX}], got "
            f"{payload[TOP_P]}"
        )
    if TEMPERATURE in payload:
        assert payload[TEMPERATURE] >= TEMPERATURE_MIN, (
            f"{TEMPERATURE} must be a float with value at least {TEMPERATURE_MIN}, got "
            f"{payload[TEMPERATURE]}."
        )
    if DO_SAMPLE in payload:
        assert (
            type(payload[DO_SAMPLE]) == bool
        ), f"{DO_SAMPLE} must be a boolean, got {payload[DO_SAMPLE]}."


def _update_num_beams(payload: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, int]]:
    """Add num_beans to the payload if missing and num_return_sequences is present."""
    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS not in payload:
        payload[NUM_BEAMS] = payload[NUM_RETURN_SEQUENCES]
    return payload


def transform_fn(model_objs: list, input_data: bytes, content_type: str, accept: str) -> bytes:
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        model_objs (list): tokenizer, model
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.
    Returns:
        obj: a byte string of the prediction
    """
    tokenizer = model_objs[0]
    model = model_objs[1]
    
    if content_type == APPLICATION_X_TEXT:
        try:
            input_text = input_data.decode(STR_DECODE_CODE)
        except Exception:
            logging.exception(
                f"Failed to parse input payload. For content_type={APPLICATION_X_TEXT}, input "
                f"payload must be a string encoded in utf-8 format."
            )
            raise
        try:
            # output = text_generator(input_text)[0]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            original_outputs = model.generate(input_ids,
                                              GenerationConfig(max_new_tokens=200)
                                             )
            output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        except Exception:
            logging.exception("Failed to do inference")
            raise
    # elif content_type == APPLICATION_JSON:
    #     try:
    #         payload = json.loads(input_data)
    #     except Exception:
    #         logging.exception(
    #             f"Failed to parse input payload. For content_type={APPLICATION_JSON}, input "
    #             f"payload must be a json encoded dictionary with keys {ALL_PARAM_NAMES}."
    #         )
    #         raise
    #     _validate_payload(payload)
    #     payload = _update_num_beams(payload)
    #     if SEED in payload:
    #         set_seed(payload[SEED])
    #         del payload[SEED]
    #     try:
    #         model_output = text_generator(**payload)
    #         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    #         original_outputs = model.generate(input_ids,
    #                                           GenerationConfig(max_new_tokens=200)
    #                                          )
    #         model_output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    #         output = {GENERATED_TEXTS: [x[GENERATED_TEXT] for x in model_output]}
    #     except Exception:
    #         logging.exception("Failed to do inference")
    #         raise
    else:
        raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
    if accept.endswith(VERBOSE_EXTENSION):
        accept = accept.rstrip(VERBOSE_EXTENSION)  # Verbose and non-verbose response are identical
    return encoder.encode(output, accept)
