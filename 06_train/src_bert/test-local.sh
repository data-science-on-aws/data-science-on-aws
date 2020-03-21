#python copy_data_locally.py

SM_CHANNEL_TRAIN=data/train SM_CHANNEL_VALIDATION=data/validation SM_MODEL_DIR=. python bert_reviews.py --model-type=distilbert --model-name=distilbert-base-cased --distributed=False
