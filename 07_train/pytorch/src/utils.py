import argparse
import json
import logging
import glob
import os
import sys
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

import pandas as pd
import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaForSequenceClassification

import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig
from smdebug import modes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Has to be called 'model.pth'
MODEL_NAME = 'model.pth'
PRE_TRAINED_MODEL_NAME = 'roberta-base'

DATA_COLUMN = 'review_body'
LABEL_COLUMN = 'sentiment'
LABEL_VALUES = [-1, 0, 1]
CLASS_NAMES = ['negative', 'neutral', 'positive']

LABEL_MAP = {}
for (i, label) in enumerate(LABEL_VALUES):
    LABEL_MAP[label] = i
    
    
def setDebuggerSaveConfig():  
    smd.SaveConfig(
        mode_save_configs={
            smd.modes.TRAIN: smd.SaveConfigMode(save_interval=1),
            smd.modes.EVAL: smd.SaveConfigMode(save_interval=1),
            smd.modes.PREDICT: smd.SaveConfigMode(save_interval=1),
            smd.modes.GLOBAL: smd.SaveConfigMode(save_interval=1)
        }
    )
    
def parse_args():

    parser = argparse.ArgumentParser()
    ###### CLI args
    parser.add_argument('--train_batch_size', 
                        type=int, 
                        default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--validation_batch_size', 
                        type=int, 
                        default=128, metavar='N',
                        help='input batch size for validation (default: 128)') 
    
    parser.add_argument('--test_batch_size', 
                        type=int, 
                        default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', 
                        type=float, 
                        default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--seed', 
                        type=int, 
                        default=42, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--log_interval', 
                        type=int, 
                        default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--backend', 
                        type=str, 
                        default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    
    parser.add_argument('--max_seq_len', 
                        type=int, 
                        default=64, 
                        help='max sequence length of input tokens')
    
    parser.add_argument("--model_name", 
                        type=str, 
                        default=MODEL_NAME, 
                       help='Model name')
    
    parser.add_argument('--enable_sagemaker_debugger', 
                        type=eval, 
                        default=False)
    
    parser.add_argument('--run_validation', 
                        type=eval,
                        default=False)  
    
    parser.add_argument('--run_test', 
                        type=eval, 
                        default=False)    
    
    parser.add_argument('--run_sample_predictions', 
                        type=eval, 
                        default=False)
    
    parser.add_argument('--enable_checkpointing', 
                        type=eval, 
                        default=False) 
    
    parser.add_argument('--checkpoint_base_path', 
                        type=str, 
                        default='/opt/ml/checkpoints')
    
    parser.add_argument('--train_steps_per_epoch',
                        type=int,
                        default=None)

    ###### Container environment   
    parser.add_argument('--hosts', 
                        type=list, 
                        default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--current_host', 
                        type=str, 
                        default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--model_dir', 
                        type=str, 
                        default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--train_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--validation_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_VALIDATION'])
    
    parser.add_argument('--test_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_TEST'])
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default=os.environ['SM_OUTPUT_DIR'])
    
    parser.add_argument('--num_gpus', 
                        type=int, 
                        default=os.environ['SM_NUM_GPUS'])
    
    # Debugger Args
    parser.add_argument("--save-frequency", 
                        type=int, 
                        default=10, 
                        help="frequency with which to save steps")
    
    parser.add_argument("--smdebug_path",
                        type=str,
                        help="output directory to save data in",
                        default="/opt/ml/output/tensors",)
    
    parser.add_argument("--hook-type",
                        type=str,
                        choices=["saveall", "module-input-output", "weights-bias-gradients"],
                        default="saveall",)

    return parser.parse_args()



class ReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_seq_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_seq_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )

        return encoding['input_ids'].flatten(), torch.tensor(target, dtype=torch.long)

    
def create_list_input_files(path):
    input_files = glob.glob('{}/*.tsv'.format(path))
    print(input_files)
    return input_files

    
def create_data_loader(path, tokenizer, max_seq_len, batch_size):
    logger.info("Get data loader")

    df = pd.DataFrame(columns=['sentiment', 'review_body'])
    
    input_files = create_list_input_files(path)

    for file in input_files:
        df_temp = pd.read_csv(file, 
                              sep='\t', 
                              usecols=['sentiment', 'review_body']
                             )
        df = df.append(df_temp)
        
    print(len(df))
    print('df[sentiment]: {}'.format(df['sentiment']))
    
    df['sentiment'] = df.sentiment.apply(lambda sentiment: LABEL_MAP[sentiment])
    print('df[sentiment] after LABEL_MAP: {}'.format(df['sentiment']))
    print(df.head())
    
    ds = ReviewDataset(
        reviews=df.review_body.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    ), df



# TODO: need to put saved config.json in code/ folder
def save_transformer_model(model, model_dir):
    path = '{}/transformer'.format(model_dir)
    os.makedirs(path, exist_ok=True)                              
    logger.info('Saving Transformer model to {}'.format(path))
    model.save_pretrained(path)


# Needs to saved in model_dir root folder
def save_pytorch_model(model, model_dir):
    # path = '{}/pytorch'.format(model_dir)
    os.makedirs(model_dir, exist_ok=True) 
    logger.info('Saving PyTorch model to {}'.format(model_dir))
    save_path = os.path.join(model_dir, MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    
def load_transformer_model(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = RobertaConfig.from_json_file('{}/config.json'.format(model_dir))
    model = RobertaForSequenceClassification.from_pretrained(model_dir, config=config)
    model = model.to(device)
    return model

def load_pytorch_model(model_dir):
    model_path = '{}/{}'.format(model_dir, MODEL_NAME)
    model = RobertaForSequenceClassification()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))  
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))    
    return model
    

def train_model(model,
                train_data_loader,
                df_train,
                val_data_loader, 
                df_val,
                args):

    
    #create smdebug hook
    setDebuggerSaveConfig()
    hook = smd.Hook.create_from_json_file()   
    hook.register_module(model)
    
    loss_function = nn.CrossEntropyLoss()
    hook.register_loss(loss_function)
    
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    
    if args.enable_sagemaker_debugger:
        print('Enable SageMaker Debugger.')

    for epoch in range(args.epochs):
        print('EPOCH -- {}'.format(epoch))

        train_correct = 0
        train_total = 0
        
        for i, (sent, label) in enumerate(train_data_loader):
            hook.set_mode(modes.TRAIN)
            model.train()
            optimizer.zero_grad()
            sent = sent.squeeze(0)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)
            
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            
            if i%10 == 0:
                train_total += label.size(0)
                train_correct += (predicted.cpu() == label.cpu()).sum()
                accuracy = 100.00 * train_correct.numpy() / train_total
                print('[epoch: {0} / step: {1}] train_loss: {2:.2f} - train_acc: {3:.2f}%'.format(epoch, i, loss.item(), accuracy))
                        
            if args.run_validation:
                if i%10 == 0:
                    hook.set_mode(modes.EVAL)
                    print('RUNNING VALIDATION:')
                    correct = 0
                    total = 0
                    model.eval()
                    input_tokens = np.array([])
                    for sent, label in val_data_loader:
                        sent = sent.squeeze(0)
                        print('sent: {}'.format(sent))
                        print('sent type: {}'.format(type(sent)))
                        
                        if torch.cuda.is_available():
                            sent = sent.cuda()
                            label = label.cuda()
                        output = model.forward(sent)[0]
                        _, predicted = torch.max(output.data, 1)
                        
                        total += label.size(0)
                        correct += (predicted.cpu() == label.cpu()).sum()
                
                    accuracy = 100.00 * correct.numpy() / total
                    print('[epoch: {0} / step: {1}] val_loss: {2:.2f} - val_acc: {3:.2f}%'.format(epoch, i, loss.item(), accuracy))
                    
                    if hook.get_collections()['all'].save_config.should_save_step(modes.EVAL, hook.mode_steps[modes.EVAL]):
                        hook._write_raw_tensor_simple("input_tokens", input_tokens)

    print('TRAINING COMPLETED.')
    return model


            #record input tokens
#            input_tokens = np.array([])
#            for example_id in example_ids.asnumpy().tolist():
#                array = np.array(dev_dataset[example_id][0].tokens, dtype=np.str)
#                array = array.reshape(1, array.shape[0])
#                input_tokens = np.append(input_tokens, array)

#            if hook.get_collections()['all'].save_config.should_save_step(modes.EVAL, hook.mode_steps[modes.EVAL]):  
#                hook._write_raw_tensor_simple("input_tokens", input_tokens)