import os, argparse, pickle
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re

def load_dataset(path, sep):
    data = pd.read_csv(path, sep=sep)

    labels = data['is_positive_sentiment']
    features = data.drop(['is_positive_sentiment'], axis=1)

    return features, labels


def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgboost_reviews.model'))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--num-round', type=int, default=1)   
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
   
    args, _ = parser.parse_known_args()   
    objective  = args.objective    
    max_depth  = args.max_depth
    num_round  = args.num_round
    model_dir  = args.model_dir
    train_data   = args.train_data
    validation_data = args.validation_data
    
    # Load transformed features (is_positive_sentiment, f0, f1, ...)    
    X_train, y_train = load_dataset(train_data, ',')
    X_validation, y_validation = load_dataset(validation_data, ',')
                
    classifier = XGBClassifier(objective=objective, 
                               num_round=num_round,
                               max_depth=max_depth)
    
    classifier.fit(X_train, y_train)

    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
    classifier.save(os.path.join(model_dir, 'xgboost_review.model'))
    
    auc = classifier.score(X_validation, y_validation)
    print("AUC ", auc)
   
    preds = classifier.predict(X_validation)
    print('Accuracy: ', accuracy_score(y_validation, preds))
    print('Precision: ', precision_score(y_validation, preds, average=None))

    model_restored = model_fn('.')
    preds_restored = model_restored.predict(X_validation)

    print('Accuracy: ', accuracy_score(y_validation, preds_restored))
    print('Precision: ', precision_score(y_validation, preds_restored, average=None))
