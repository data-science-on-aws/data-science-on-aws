#!/bin/bash

pip install scikit-learn==0.23.1
pip install tensorflow==2.3.1
pip install transformers==3.5.1

mkdir -p ./output

python evaluate_model_metrics.py --hosts=algo-1,algo-2 --current-host=algo-1 --input-data=./ --output-data=./output --max-seq-length=64

echo "Transformed data is in ./output/"
