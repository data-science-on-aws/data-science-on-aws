import os
import argparse
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
import xgboost as xgb
from xgboost import XGBClassifier

# Note:  header=None
def load_dataset(path, sep, header):
    data = pd.read_csv(path, sep=sep, header=header)

    labels = data.iloc[:,0]
    features = data.drop(data.columns[0], axis=1)
    
    if header==None:
        # Adjust the column names after dropped the 0th column above
        # New column names are 0 (inclusive) to len(features.columns) (exclusive)
        new_column_names = list(range(0, len(features.columns)))
        features.columns = new_column_names

    return features, labels


def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = pkl.load(open(model_path, 'rb'))
    return model


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
                
    import xgboost as xgb
    from xgboost import XGBClassifier

    model = XGBClassifier(objective=objective,
                          num_round=num_round,
                          max_depth=max_depth)

    model.fit(X_train, y_train)

    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
    # Need to save with joblib or pickle.  `xgb.save_model()` does not save feature_names

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost-model')
    pkl.dump(model, open(model_path, 'wb'))
    print('Wrote model to {}'.format(model_path))

    model_restored = model_fn(model_dir)
    preds_validation = model_restored.predict(X_validation)

    auc = model_restored.score(X_validation, y_validation)
    print('Validation AUC: ', auc)

    preds_validation = model_restored.predict(X_validation)
    print('Validation Accuracy: ', accuracy_score(y_validation, preds_validation))
    print('Validation Precision: ', precision_score(y_validation, preds_validation, average=None))
    
    print(classification_report(y_validation, preds_validation))
