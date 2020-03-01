import os
import pickle as pkl
import numpy as np

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

def model_fn(model_dir):
    model_file = model_dir + '/model.bin'
    model = pkl.load(open(model_file, 'rb'))
    return model

def output_fn(prediction, accept):
    
    pred_array_value = np.array(prediction)
    pred_value = int(pred_array_value[0])
    
    return worker.Response(str(pred_value), accept, mimetype=accept)
