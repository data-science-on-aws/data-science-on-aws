#!/bin/bash

mkdir -p ./output/scikit/bert/train
mkdir -p ./output/scikit/bert/validation
mkdir -p ./output/scikit/bert/test

python preprocess-scikit-text-to-bert.py --hosts=algo-1,algo-2 --current-host=algo-1 --input-data=./data --output-data=./output/scikit
