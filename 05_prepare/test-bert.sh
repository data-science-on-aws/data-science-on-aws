#!/bin/bash

mkdir -p ./data/output/bert/train
mkdir -p ./data/output/bert/validation
mkdir -p ./data/output/bert/test

python preprocess-scikit-text-to-bert.py --hosts=algo-1,algo-2 --current-host=algo-1 --input-data=./data --output-data=./data/output

