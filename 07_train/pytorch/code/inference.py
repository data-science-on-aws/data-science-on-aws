import json
import sys
import logging
import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###################################
### VARIABLES 
###################################

# Needs to be called 'model.pth' as per 
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/6936c08581e26ff3bac26824b1e4946ec68ffc85/src/sagemaker_pytorch_serving_container/torchserve.py#L45
MODEL_NAME = 'model.pth'

PRE_TRAINED_MODEL_NAME = 'roberta-base'
MAX_SEQ_LEN = 64

JSON_CONTENT_TYPE = 'application/json'

DATA_COLUMN = 'review_body'
LABEL_COLUMN = 'sentiment'
LABEL_VALUES = [-1, 0, 1]
CLASS_NAMES = ['negative', 'neutral', 'positive']

LABEL_MAP = {}
for (i, label) in enumerate(LABEL_VALUES):
    LABEL_MAP[label] = i

# Load Hugging Face Tokenizer
TOKENIZER = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)  


###################################
### SAGEMKAER LOAD MODEL FUNCTION 
###################################   

# You need to put in config.json from saved fine-tuned Hugging Face model in code/ 
# Reference it in the inference container at /opt/ml/model/code
def model_fn(model_dir):
    model_path = '{}/{}'.format(model_dir, MODEL_NAME) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = RobertaConfig.from_json_file('/opt/ml/model/code/config.json')
    model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
    model.to(device)
    return model

###################################
### SAGEMKAER PREDICT FUNCTION 
###################################   

def predict_fn(input_data, model):
    model.eval()
  
    review_text = input_data['review_body']
    
    encoded_review = TOKENIZER.encode_plus(
        review_text,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True)
    
    input_ids = encoded_review['input_ids']
    print('input_ids: {}'.format(input_ids))
    print('type input_ids: {}'.format(type(input_ids)))
    attention_mask = encoded_review['attention_mask']
    print('attention_mask: {}'.format(attention_mask))
    print('type attention_mask: {}'.format(type(attention_mask)))
        
    output = model(input_ids, attention_mask)
    print('output: {}'.format(output))
    print('type output: {}'.format(type(output)))

    # output is a tuple: 
    # output: (tensor([[-1.9840, -0.9870,  2.8947]], grad_fn=<AddmmBackward>),
    # for torch.max() you need to pass in the tensor, output[0]   
    _, prediction = torch.max(output[0], dim=1)
    
    print('prediction: {}'.format(prediction))
    print('type prediction: {}'.format(type(prediction)))
    
    pred = prediction.item()
    
    print('pred: {}'.format(pred))
    print('type pred: {}'.format(type(pred)))
    
    # print('LABEL_MAP: {}'.format(LABEL_MAP))
    print('CLASS_NAMES: {}'.format(CLASS_NAMES))
    
    # print('LABEL_MAP[pred]: {}'.format(LABEL_MAP[pred]))
    # print('CLASS_NAMES[LABEL_MAP[pred]]: {}'.format(CLASS_NAMES[LABEL_MAP[pred]]))
    
    return CLASS_NAMES[pred]

#LABEL_MAP = {-1:0, 0:1, 1:2}

###################################
### SAGEMKAER MODEL INPUT FUNCTION 
################################### 

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):  
    if content_type == JSON_CONTENT_TYPE:
        data = json.loads(serialized_input_data)
        return data
    else:
        pass

###################################
### SAGEMKAER MODEL OUTPUT FUNCTION 
################################### 

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)