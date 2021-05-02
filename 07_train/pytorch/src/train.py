import argparse
import pprint
import json
import logging
import os
import sys
import pandas as pd
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader


from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import RobertaForSequenceClassification

import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig
from smdebug import modes

from utils import create_data_loader, train_model, parse_args, save_pytorch_model, save_transformer_model

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


if __name__ == '__main__':
    
    ###### Parse ARGS
    args = parse_args()
    print('Loaded arguments:')
    print(args)

    ###### Get Environment Variables
    env_var = os.environ 
    print('Environment variables:')
    pprint.pprint(dict(env_var), width = 1) 
    
    print('SM_TRAINING_ENV {}'.format(env_var['SM_TRAINING_ENV']))
    sm_training_env_json = json.loads(env_var['SM_TRAINING_ENV'])

    ###### Check if Training Master
    is_master = sm_training_env_json['is_master']
    print('is_master {}'.format(is_master))
    
    if is_master:
        checkpoint_path = args.checkpoint_base_path
    else:
        checkpoint_path = '/tmp/checkpoints'        
    print('checkpoint_path {}'.format(checkpoint_path))
    
    ###### Check if distributed training
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device('cuda' if use_cuda else 'cpu')
     
    if is_distributed:
        ###### Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))
    
    ###### Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed) 

    
    ###### INSTANTIATE MODEL
    tokenizer = None
    config = None
    model = None
    
    successful_download = False
    retries = 0
    
    while (retries < 5 and not successful_download):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
            
            config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                   num_labels=len(CLASS_NAMES),
                                                   id2label={
                                                       0: -1,
                                                       1: 0,
                                                       2: 1,
                                                   },
                                                   label2id={
                                                       -1: 0,
                                                       0: 1,
                                                       1: 2,
                                                   })
            config.output_attentions=True
            model = RobertaForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, 
                                                                     config=config)
            model.to(device)
            successful_download = True
            print('Sucessfully downloaded after {} retries.'.format(retries))
        
        except:
            retries = retries + 1
            random_sleep = random.randint(1, 30)
            print('Retry #{}.  Sleeping for {} seconds'.format(retries, random_sleep))
            time.sleep(random_sleep)
 
    if not tokenizer or not model or not config:
         print('Not properly initialized...')
            
    ###### CREATE DATA LOADERS
    train_data_loader, df_train = create_data_loader(args.train_data, tokenizer, args.max_seq_len, args.train_batch_size)
    val_data_loader, df_val = create_data_loader(args.validation_data, tokenizer, args.max_seq_len, args.validation_batch_size)
    
    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_data_loader.sampler), len(train_data_loader.dataset),
        100. * len(train_data_loader.sampler) / len(train_data_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(val_data_loader.sampler), len(val_data_loader.dataset),
        100. * len(val_data_loader.sampler) / len(val_data_loader.dataset)
    )) 
       
    # model_dir = os.environ['SM_MODEL_DIR']
    print('model_dir: {}'.format(args.model_dir))
    
    print('model summary: {}'.format(model))
        
    ###### START TRAINING

    model = train_model(model,
                        train_data_loader,
                        df_train,
                        val_data_loader, 
                        df_val,
                        args)
    
    save_transformer_model(model, args.model_dir)
    save_pytorch_model(model, args.model_dir)