#Utils for Receipt Demo
import boto3
import sagemaker
import sys
import os
import re
import numpy as np
import pandas as pd
import subprocess
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
import gzip
from io import BytesIO
import zipfile
import random
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from matplotlib import pylab
import nltk 
import spacy
import en_core_web_lg
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer 
import string
import pickle
from nltk.corpus import stopwords 
from itertools import combinations 
import operator
from io import BytesIO
import tarfile
import seaborn as sns

def setup_env(configs, global_vars):
   
    sess = sagemaker.Session()
    
    role = get_execution_role()

    AWS_REGION = configs['aws_region']
    s3 = boto3.resource('s3')

    s3_bucket = s3.Bucket(configs['bucket_name'])

    if s3_bucket.creation_date == None:
    # create S3 bucket because it does not exist yet
        print('Creating S3 bucket {}.'.format(bucket))
        resp = s3.create_bucket(
            ACL='private',
            Bucket=bucket
        )
    else:
        print('Bucket already exists')
        
    global_vars['role'] = role
    global_vars['sess'] = sess
    global_vars['s3'] = s3
    global_vars['s3_bucket'] = s3_bucket
    
    
    #set up textract
    textract = boto3.client('textract')
    
    global_vars['textract'] = textract

    
    return global_vars


def load_or_save_record_meta_data(textract_data=None, load_or_save='load'):
    
    tmp_filename = 'tmp.pickle'
    if load_or_save == 'load':
        with open(tmp_filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

        
    if load_or_save == 'save':
        with open(tmp_filename, 'wb') as handle:
            pickle.dump(textract_data, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        print('Saved to {}'.format(tmp_filename))
        return textract_data
    
def inspect_dataset(textract_data):
    
    print('Total Records {}'.format(len(textract_data)))
    
def preprocess_text(text):
    
    processed_text = text.lower().strip().replace(',','').replace(':',''). \
                    replace('?','').replace('000','').replace('>','').replace('<','').replace('!',''). \
                    replace(')','').replace('(','').replace('#','')
    
    #we have some oddities
    processed_text =  ' '.join(i for i in processed_text.split(' ') if not i.endswith('pm'))
    processed_text  = ' '.join(i for i in processed_text.split(' ') if not i.endswith('am'))
    
    return processed_text
    
def visualize_detection(image_path, bounding_boxes):

    local_path = image_path

    with open(local_path, 'rb') as f:
        img_file = f.read()
        img_file = bytearray(img_file)
        ne = open('n.txt','wb')
        ne.write(img_file)

   
    img=mpimg.imread(local_path)
    height = img.shape[0]
    width = img.shape[1]
    height, width, depth = img.shape
    dpi = 80
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize = figsize)
    plt.imshow(img)
    
    colors = dict()

    idx = 0
    for det in bounding_boxes:
        (x0, y0, x1, y1, word) = det
        try:
            colors[idx] = (random.random(), random.random(), random.random())

            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[idx],
                                 linewidth=3.5)
            plt.gca().add_patch(rect)

            plt.gca().text(xmin, ymin ,
                            '{}'.format(word),
                            bbox=dict(facecolor=colors[idx], alpha=0.5),
                                    fontsize=8, color='white')
            idx += 1
        except:
            pass
    plt.show()
