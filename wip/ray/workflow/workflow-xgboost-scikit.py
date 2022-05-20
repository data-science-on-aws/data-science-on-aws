#import ray
from ray import workflow
from typing import List

import ray
from xgboost_ray import RayXGBClassifier, RayParams
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

seed = 42

@workflow.step
def read_data():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y

@workflow.step
def preprocessing(data):
    X, y = data 

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        train_size=0.25, 
        random_state=42
    )

    return X_train, y_train, X_test, y_test

@workflow.step
def train(preprocessed_data):
    X_train, y_train, X_test, y_test = preprocessed_data 

    classifier = RayXGBClassifier(
        n_jobs=4,  # In XGBoost-Ray, n_jobs sets the number of actors
        random_state=seed
    )

# scikit-learn API will automatically convert the data
# to RayDMatrix format as needed.
# You can also pass X as a RayDMatrix, in which case
# y will be ignored.

    classifier.fit(X_train, y_train)

    return classifier


@workflow.step
def validate(classifier, preprocessed_data):
    X_train, y_train, X_test, y_test = preprocessed_data 

    pred_ray = classifier.predict(X_test)
    print(pred_ray)

    pred_proba_ray = classifier.predict_proba(X_test)
    print(pred_proba_ray)

    return "TODO"


# It is also possible to pass a RayParams object
# to fit/predict/predict_proba methods - will override
# n_jobs set during initialization

#clf.fit(X_train, y_train, ray_params=RayParams(num_actors=2))

#pred_ray = clf.predict(X_test, ray_params=RayParams(num_actors=2))
#print(pred_ray)

#ray.init(address="auto", storage=None)

# Initialize workflow storage.
workflow.init()

# Setup the workflow.
data = read_data.step()

preprocessed_data = preprocessing.step(data)

classifier = train.step(preprocessed_data)

metrics = validate.step(classifier, preprocessed_data)

# Execute the workflow and print the result.
print('**** {} ****'.format(metrics.run()))

# The workflow can also be executed asynchronously.
# print(ray.get(output.run_async()))
