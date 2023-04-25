from djl_python import Input, Output
from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import subprocess
import sys

hf_pipeline = None

def load_model(properties):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    peft_model_id = "google/flan-t5-small_LORA_SEQ_2_SEQ_LM"
    #peft_model_id = properties.get("model_id")
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    hf_pipeline = pipeline(task="summarization", model=model, tokenizer=tokenizer, device=device)
    
    return hf_pipeline


def run_inference(hf_pipeline, data, params):
    
    outputs = hf_pipeline(data, **params)
    
    return outputs


def handle(inputs: Input):
    global hf_pipeline
    if not hf_pipeline:
        hf_pipeline = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()

    inputs = data["inputs"]
    inputs = ["summarize: " + inp for inp in inputs]
    
    params = data.get("parameters", {})

    outputs = run_inference(hf_pipeline, inputs, params)
    result = {"outputs": outputs}
    return Output().add_as_json(result)