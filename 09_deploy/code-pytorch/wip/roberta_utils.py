import json
import torch
import sagemaker
import time

from transformers import RobertaModel, RobertaTokenizer

CLASS_NAMES = ['negative', 'neutral', 'positive']
MODEL_NAME = 'model.pt'


###################################
### SAVE/LOAD MODEL CHECKPOINTS 
###################################

def save_checkpoint(model, optimizer, model_dir):
    timestamp = int(time.time())
    checkpoint_name = '[]-[]'.format(MODEL_NAME, timestamp)
    path = os.path.join(model_dir, checkpoint_name)
    torch.save(model.state_dict(), optimizer.state_dict(), path)
    print('Saved model checkpoint: {}'.format(path))
    return path
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = SentimentClassifier(len(CLASS_NAMES))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    print('Loaded model checkpoint: {}'.format(path))
    return model, optimizer

###################################
### SAVE/LOAD MODEL 
###################################

def save_model(model, model_dir):
    path = os.path.join(model_dir, MODEL_NAME)
    torch.save(model.state_dict(), path)
    print('Saved model to path: {}'.format(path))
    return path
    
def load_model(model_dir):
    path = os.path.join(model_dir, MODEL_NAME)
    model = SentimentClassifier(len(CLASS_NAMES))
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(path, map_location='cuda:0'))  
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(path, map_location=device))
    print('Loaded model {} with device {} and map_location {}'.format(path, device, map_location))
    model.to(device)
    return model
    
    