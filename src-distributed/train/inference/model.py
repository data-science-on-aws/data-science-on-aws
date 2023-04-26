from djl_python import Input, Output
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, GenerationConfig
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
from peft import PeftModel, PeftConfig

hf_pipeline = None
model = None
tokenizer = None

def load_pipeline(properties):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    peft_model_id = "google/flan-t5-small_LORA_SEQ_2_SEQ_LM"
    #peft_model_id = properties.get("model_id")
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            
    hf_pipeline = pipeline(task="summarization", model=model, tokenizer=tokenizer, device=device)
    return hf_pipeline


def load_model_tokenizer(properties):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    peft_model_id = "google/flan-t5-small_LORA_SEQ_2_SEQ_LM"
    #peft_model_id = properties.get("model_id")
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    return model, tokenizer


def run_inference_pipeline(hf_pipeline, data, params):
    
    outputs = hf_pipeline(data, **params)
    
    return outputs


def run_inference_model_tokenizer(model, tokenizer, data, params):
    response = None
    inputs = tokenizer(data, return_tensors='pt')
    input_ids = inputs["input_ids"]
    try:
        response = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))     
        print("response succeeded with generation_config!")
    except:
        print("response failed with generation_config")
    
    try:
        response = tokenizer.decode(
            model.generate(
               input_ids=input_ids, 
               max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
        )        
        print("response succeeded with regular generate!")
    except:
        print("response failed with regular generate")

    return response


def handle(inputs: Input):
    global hf_pipeline
    if not hf_pipeline:
        try:
            hf_pipeline = load_pipeline(inputs.get_properties())
            print("handle succeeded with hf_pipeline!")
        except:
            print("handle failed with hf_pipeline")

    global model
    global tokenizer
    if not model:
        try:
            model, tokenizer = load_model_tokenizer(inputs.get_properties())
            print("handle succeeded with model, tokenizer!")            
        except:
            print("handle failed with model, tokenizer")
        
    if inputs.is_empty():
        return None
    data = inputs.get_as_json()

    inputs = data["inputs"]
    inputs = ["summarize: " + inp for inp in inputs]
    
    params = data.get("parameters", {})

    outputs = None
    try:
        outputs = run_inference_pipeline(hf_pipeline, inputs, params)
        print("inference succeeded with pipeline!")        
    except:
        print("inference failed with pipeline")
    
    try:
        outputs = run_inference_model_tokenizer(model, tokenizer, inputs, params)
        print("inference succeeded with model_tokenizer!")        
    except:
        print("inference failed with model_tokenizer")
    
    result = {"outputs": outputs}
    return Output().add_as_json(result)