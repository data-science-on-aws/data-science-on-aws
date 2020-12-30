python copy_data_locally.py

SM_CHANNEL_TRAIN=feature-store/amazon-reviews/csv/balanced-tfidf-without-header/train SM_CHANNEL_VALIDATION=feature-store/amazon-reviews/csv/balanced-tfidf-without-header/validation SM_MODEL_DIR=. python xgboost_reviews.py 
# --num-rounds=10
