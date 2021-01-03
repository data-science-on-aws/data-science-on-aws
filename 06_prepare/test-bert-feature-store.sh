#!/bin/bash

pip install scikit-learn==0.23.1
#pip install pandas==1.0.5
#pip install tensorflow==2.3.1
#pip install transformers==3.5.1
#pip install sagemaker==2.22.0

mkdir -p ./output/scikit/bert/train
mkdir -p ./output/scikit/bert/validation
mkdir -p ./output/scikit/bert/test

FEATURE_GROUP_NAME=$(date "+%s")

AWS_DEFAULT_REGION=us-east-1 python preprocess-scikit-text-to-bert-feature-store.py --hosts=algo-1,algo-2 --current-host=algo-1 --input-data=./data --output-data=./output/scikit --train-split-percentage=0.90 --validation-split-percentage=0.05 --test-split-percentage=0.05 --balance-dataset=True --max-seq-length=64 --feature-group-name=feature-group-$FEATURE_GROUP_NAME --feature-store-offline-prefix=feature-group-$FEATURE_GROUP_NAME

echo "Transformed data is in ./output/"
