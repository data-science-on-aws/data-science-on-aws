from djl_python import Input, Output
from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import fastertransformer as ft
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    GenerationConfig,
    pipeline
)
import os
import logging
import math
import traceback


peft_pipeline = None
model = None
tokenizer = None

def load_model(properties):
    model_name = "google/flan-t5-large"
    tensor_parallel_degree = properties.get("tensor_parallel_degree", 1)
    pipeline_parallel_degree = 1
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model in {model_location}")


    # tokenizer = T5Tokenizer.from_pretrained(model_location)
    # dtype = "fp32"
    # model = ft.init_inference(
    #     model_location, tensor_parallel_degree, pipeline_parallel_degree, dtype
    # )
    # return model, tokenizer



    # load base LLM model and tokenizer
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # Load the LoRA/PEFT model
    peft_model = PeftModel.from_pretrained(peft_model_base, f'{model_location}/', device_map="auto")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    peft_model.eval()
    peft_model.to(device)
    
    total_params = sum(p.numel() for p in peft_model.parameters())

    logging.info(f"Peft model loaded:total_params={total_params}::")

    peft_pipe = pipeline(task="summarization", model=peft_model, tokenizer=tokenizer, device=device)
    
    return peft_pipe, peft_model, tokenizer



def run_pipeline(peft_pipeline, tokenizer, data):
    try:
        logging.info(f'run_pipeline():Starting')
        input_sentences = data["inputs"]
        params = data["parameters"]
        logging.info(f"run_pipeline():PEFT:loaded:{type(input_sentences)}::input_sentences={input_sentences}::params={params}::")
        
        #diag=['Amazon.com is the best ']
        prompt = f'Summarize the following conversation.\n\n{input_sentences}\n\nSummary:'
        # - max_new_tokens` and `max_length` h
        params = {'max_length':200, 'num_beams':1}
        peft_model_text_output = peft_pipeline(prompt, **params)

        logging.info(f'run_pipeline():PEFT:Prompt:\n--------------------------\n{prompt}\n--------------------------')
        logging.info(f'run_pipeline():PEFT: model summary: {peft_model_text_output}')
        result = {"outputs": peft_model_text_output}
        return result
    
    except:
        err_str = traceback.format_exc()
        logging.info(f"error in run_pipeline()::{err_str}")  
        return {"outputs": err_str}
        
def run_model_inference(model, tokenizer, data):
    
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        input_sentences = data["inputs"]
        params = data["parameters"]
        logging.info(f"run_model_inference():PEFT:loaded:{type(input_sentences)}::input_sentences={input_sentences}::params={params}::")

        prompt = f'Summarize the following conversation.\n\n{input_sentences}\n\nSummary:'
        input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids

        model_outputs = model.generate(input_ids=input_ids) #, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)) 
        # - input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)) #gen_config)
        logging.info(f"run_model_inference():PEFT:OUTPUT:peft_model_outputs:{model_outputs}:")

        model_text_output = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        logging.info(f"run_model_inference():PEFT:model_text_output={model_text_output}:")

        result = {"outputs": model_text_output}
        return result
    except:
        err_str = traceback.format_exc()
        logging.info(f"error in hardcode::{err_str}")  
        return {"outputs": err_str}
    
    

def handle(inputs: Input):
    """
    inputs: Contains the configurations from serving.properties
    """
    global peft_pipeline, model, tokenizer
    # - peft_pipe, peft_model, tokenizer
    if not peft_pipeline:
        peft_pipeline, model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    
    data = inputs.get_as_json()
    
    outputs = run_pipeline(peft_pipeline, tokenizer, data)
    
    run_model_inference(model, tokenizer, data) 
                      
    return Output().add_as_json(outputs)

