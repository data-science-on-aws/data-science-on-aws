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

# Note:  header=None
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
    auc = round(metrics.roc_auc_score(y_validation, preds_validation), 4)
    print('AUC is ' + repr(auc))
