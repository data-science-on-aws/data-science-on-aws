import json
import sys
import logging
import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###################################
### VARIABLES 
###################################
max_seq_length = 64

classes = [1, 2, 3, 4, 5]

#LABEL_MAP = {-1:0, 0:1, 1:2}
# LABEL_MAP = {}
# for (i, label) in enumerate(LABEL_VALUES):
#     LABEL_MAP[label] = i

# Load Hugging Face Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  


###################################
### SAGEMKAER LOAD MODEL FUNCTION 
###################################   

# You need to put in config.json from saved fine-tuned Hugging Face model in code/ 
# Reference it in the inference container at /opt/ml/model/code
# The model needs to be called 'model.pth' per https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/6936c08581e26ff3bac26824b1e4946ec68ffc85/src/sagemaker_pytorch_serving_container/torchserve.py#L45

def model_fn(model_dir):
    config = DistilBertConfig.from_json_file('/opt/ml/model/code/config.json')
    
    model_path = '{}/{}'.format(model_dir, 'model.pth') 
    model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model


###################################
### SAGEMKAER PREDICT FUNCTION 
###################################   

def predict_fn(input_data, model):
    model.eval()

    print('input_data: {}'.format(input_data))

    data_json = json.loads(input_data)
    print('data_json: {}'.format(data_json))

    predicted_classes = []

    for data_json_line in data_json:
        print('data_json_line: {}'.format(data_json_line))
        print('type(data_json_line): {}'.format(type(data_json_line)))

        review_body = data_json_line['review_body']
        print("""review_body: {}""".format(review_body))

        encode_plus_token = tokenizer.encode_plus(
            review_body,
            max_length=max_seq_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True)

        input_ids = encode_plus_token['input_ids']
        attention_mask = encode_plus_token['attention_mask']

        output = model(input_ids, attention_mask)
        print('output: {}'.format(output))

        # output is a tuple: 
        # output: (tensor([[-1.9840, -0.9870,  2.8947]], grad_fn=<AddmmBackward>),
        # for torch.max() you need to pass in the tensor, output[0]   
        _, prediction = torch.max(output[0], dim=1)

        predicted_class_idx = prediction.item()
        predicted_class = classes[predicted_class_idx]
        print('predicted_class: {}'.format(predicted_class))

        prediction_dict = {}
        prediction_dict['predicted_label'] = predicted_class

        jsonline = json.dumps(prediction_dict)
        print('jsonline: {}'.format(jsonline))

        predicted_classes.append(jsonline)
        print('predicted_classes in the loop: {}'.format(predicted_classes))

    predicted_classes_jsonlines = '\n'.join(predicted_classes)
    print('predicted_classes_jsonlines: {}'.format(predicted_classes_jsonlines))
    print('type(predicted_classes_jsonlines): {}'.format(type(predicted_classes_jsonlines)))

    predicted_classes_jsonlines_dump = json.dumps(predicted_classes_jsonlines)
    print('predicted_classes_jsonlines_dump: {}'.format(predicted_classes_jsonlines_dump))
    print('type(predicted_classes_jsonlines_dump): {}'.format(type(predicted_classes_jsonlines_dump)))

    return predicted_classes_jsonlines_dump


###################################
### SAGEMKAER MODEL INPUT FUNCTION 
################################### 

def input_fn(serialized_input_data, content_type='application/jsonlines'):  
#    if content_type == 'application/jsonlines':
    return serialized_input_data
#    raise Exception('Requested unsupported ContentType: ' + content_type)

###################################
### SAGEMKAER MODEL OUTPUT FUNCTION 
################################### 

def output_fn(prediction_output, accept='application/jsonlines'):
#    if accept == 'application/jsonlines':
    return prediction_output, accept
#    raise Exception('Requested unsupported ContentType in Accept: ' + accept)