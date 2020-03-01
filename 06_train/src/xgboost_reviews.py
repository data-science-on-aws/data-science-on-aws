import os, argparse, pickle
import xgboost as xgb
import pandas as pd
from sklearn.externals import joblib


print('XGBoost version: {}'.format(xgb.__version__))

from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]
    
import nltk
import re
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words    
    
#class NumberSelector(BaseEstimator, TransformerMixin):
#    def __init__(self, field):
#        self.field = field
#    def fit(self, X, y=None):
#        return self
#    def transform(self, X):
#        return X[[self.field]]

def load_dataset(path, sep):
    # Load dataset
    data = pd.read_csv(path, sep=sep)

    # Process dataset
    labels = data['is_positive_sentiment']
    features = data.drop(['is_positive_sentiment'], axis=1)

    return features, labels

def model_fn(model_dir):
#    model = xgb.Booster()
#    model.load_model(os.path.join(model_dir, 'xgboost_reviews.model'))
#    return model

    pipeline = joblib.load(os.path.join(model_dir, 'xgboost_reviews_pipeline.pkl'))
    return pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--model-dir', type=str,) # default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str,) # default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str,) # default=os.environ['SM_CHANNEL_VALIDATION'])
   
    args, _ = parser.parse_known_args()
#    max_depth  = args.max_depth
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    X_train, y_train = load_dataset(os.path.join(training_dir, 'train.csv'), ',')
    X_eval, y_eval = load_dataset(os.path.join(validation_dir, 'validation.csv'), ',')
    
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from xgboost import XGBClassifier

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('body', Pipeline([
                ('body_text_selector', TextSelector('review_body')),
                ('tfidf_vectorizer', TfidfVectorizer(tokenizer=Tokenizer, stop_words="english",
                         min_df=.0025, max_df=0.25, ngram_range=(1,3))),
                ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
            ])),
        ])),
        ('classifier', XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.1)),
    ])    
    
    pipeline.fit(X_train, y_train)    
    
    auc = pipeline.score(X_eval, y_eval)
    print("AUC ", auc)
    
    from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

    preds = pipeline.predict(X_eval)

    print('Accuracy: ', accuracy_score(y_eval, preds))
    print('Precision: ', precision_score(y_eval, preds, average=None))

    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
    #pipeline.save(os.path.join(model_dir, 'xgboost_review.model'))
    joblib.dump(pipeline, 'xgboost_reviews_pipeline.pkl')
    
    pipeline_restored = model_fn('.')
    preds_restored = pipeline_restored.predict(X_eval)

    print('Accuracy: ', accuracy_score(y_eval, preds_restored))
    print('Precision: ', precision_score(y_eval, preds_restored, average=None))


