#!/bin/bash

pip install scikit-learn==0.20.0
pip install tensorflow==2.1.0
pip install transformers==2.8.0

mkdir -p ./output/scikit/bert/train
mkdir -p ./output/scikit/bert/validation
mkdir -p ./output/scikit/bert/test

python preprocess-scikit-text-to-bert.py --hosts=algo-1,algo-2 --current-host=algo-1 --input-data=./data --output-data=./output/scikit --train-split-percentage=0.90 --validation-split-percentage=0.05 --test-split-percentage=0.05 --balance-dataset=True --max-seq-length=64

echo "Transformed data is in ./output/"
