import json
import sys
import logging
import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer

###################################
### VARIABLES 
###################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MODEL_NAME = 'model.pt'
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

###################################
### MODEL CLASS 
###################################

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                    id2label={
                                                        0: -1,
                                                        1: 0,
                                                        2: 1},
                                                    label2id={-1: 0,
                                                              0: 1,
                                                              1: 2})

        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

###################################
### SAGEMKAER LOAD MODEL FUNCTION 
###################################   

def model_fn(model_dir):
    model_path = '{}/{}'.format(model_dir, MODEL_NAME)
    model = SentimentClassifier(len(CLASS_NAMES))
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))  
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))    

    model.to(device)
    return model

###################################
### SAGEMKAER PREDICT FUNCTION 
###################################   

def predict_fn(input_data, model):
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)    
    review_text = input_data['review_body']
    
    encoded_review = tokenizer.encode_plus(
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
    
    _, prediction = torch.max(output, dim=1)
    
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