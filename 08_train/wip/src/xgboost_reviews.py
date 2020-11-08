import os
import argparse
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
import xgboost as xgb
from xgboost import XGBClassifier
import glob


def load_dataset(path, sep, header):
    data = pd.concat([pd.read_csv(f, sep=sep, header=header) for f in glob.glob('{}/*.csv'.format(path))], ignore_index = True)

    labels = data.iloc[:,0]
    features = data.drop(data.columns[0], axis=1)
    
    if header==None:
        # Adjust the column names after dropped the 0th column above
        # New column names are 0 (inclusive) to len(features.columns) (exclusive)
        new_column_names = list(range(0, len(features.columns)))
        features.columns = new_column_names

    return features, labels


def model_fn(model_dir):
    """
    :param: model_dir The directory where model files are stored.
    :return: a model
    """
    # IsADirectoryError: [Errno 21] Is a directory: '/opt/ml/model'
    import os
    list_dirs = os.listdir(model_dir)
    for file in dirs:
        print(file)

    model = pkl.load(open(model_dir, 'rb'))

    print(type(model))
    
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the Invoke request body into an object we can perform prediction on
    """
    """An input_fn that loads a pickled object"""
    if request_content_type == "application/json":
        pass
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass

    print(request_body)    
    return [1]


def predict_fn(input_object, model):
    """
    Perform prediction on the deserialized object, with the loaded model
    """
    return [1]


def output_fn(output, output_content_type):
    """
    Serialize the prediction result into the desired response content type
    """
    #return json.dumps({'output':output.reshape(-1).tolist()}), output_content_type
    print(output)
    return [1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--num-round', type=int, default=1)   
    parser.add_argument('--train-data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()   
    objective  = args.objective    
    max_depth  = args.max_depth
    num_round  = args.num_round
    train_data   = args.train_data
    validation_data = args.validation_data    
    model_dir  = args.model_dir
    
    # Load transformed features (is_positive_sentiment, f0, f1, ...)    
    X_train, y_train = load_dataset(train_data, ',', header=None)
    X_validation, y_validation = load_dataset(validation_data, ',', header=None)

    xgb_estimator = XGBClassifier(objective=objective,
                                  num_round=num_round,
                                  max_depth=max_depth)

    xgb_estimator.fit(X_train, y_train)

    # TODO:  use the model_dir that is passed in through args
    #        (currently SM_MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost-model')

    pkl.dump(xgb_estimator, open(model_path, 'wb'))
    print('Wrote model to {}'.format(model_path))
    
    xgb_estimator_restored = pkl.load(open(model_path, 'rb'))
    type(xgb_estimator_restored) 
    
    preds_validation = xgb_estimator_restored.predict(X_validation)
    print('Validation Accuracy: ', accuracy_score(y_validation, preds_validation))
    print('Validation Precision: ', precision_score(y_validation, preds_validation, average=None))
    
    print(classification_report(y_validation, preds_validation))

    # TODO:  Convert to preds_validation_0_or_1

    ##############
#   Note:  roc_auc is causing the following:
#   ValueError: multiclass format is not supported
#     Traceback (most recent call last):
#   File "/miniconda3/lib/python3.6/runpy.py", line 193, in _run_module_as_main
#     "__main__", mod_spec)
#   File "/miniconda3/lib/python3.6/runpy.py", line 85, in _run_code
#     exec(code, run_globals)
#   File "/opt/ml/code/xgboost_reviews.py", line 75, in <module>
#     auc = round(metrics.roc_auc_score(y_validation, preds_validation), 4)
#   File "/miniconda3/lib/python3.6/site-packages/sklearn/metrics/ranking.py", line 356, in roc_auc_score
#     sample_weight=sample_weight)
#   File "/miniconda3/lib/python3.6/site-packages/sklearn/metrics/base.py", line 74, in _average_binary_score
#     raise ValueError("{0} format is not supported".format(y_type))
 
#    auc = round(metrics.roc_auc_score(y_validation, preds_validation), 4)
#    print('AUC is ' + repr(auc))
