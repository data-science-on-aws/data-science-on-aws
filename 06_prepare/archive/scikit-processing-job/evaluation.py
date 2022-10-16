
import json
import os
import tarfile

import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

if __name__=="__main__":
    model_path = os.path.join('/opt/ml/processing/model', 'model.tar.gz')
    print('Extracting model from path: {}'.format(model_path))
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    print('Loading model')
    model = joblib.load('model.joblib')

    print('Loading test input data')
    test_features_data = os.path.join('/opt/ml/processing/test', 'test_features.csv')
    test_labels_data = os.path.join('/opt/ml/processing/test', 'test_labels.csv')

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)
    predictions = model.predict(X_test)

    print('Creating classification evaluation report')
    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_dict['accuracy'] = accuracy_score(y_test, predictions)
    report_dict['roc_auc'] = roc_auc_score(y_test, predictions)

    print('Classification report:\n{}'.format(report_dict))

    evaluation_output_path = os.path.join('/opt/ml/processing/evaluation', 'evaluation.json')
    print('Saving classification report to {}'.format(evaluation_output_path))

    with open(evaluation_output_path, 'w') as f:
        f.write(json.dumps(report_dict))