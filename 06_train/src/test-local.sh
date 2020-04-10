rm -rf model/
rm -rf output/
#SM_INPUT_DATA_CONFIG={\"train\":{\"TrainingInputMode\":\"Pipe\"}} 
SM_CURRENT_HOST=blah SM_NUM_GPUS=0 SM_HOSTS={\"hosts\":\"blah\"} SM_MODEL_DIR=model/ SM_OUTPUT_DATA_DIR=output/ SM_CHANNEL_TRAIN=data/train SM_CHANNEL_VALIDATION=data/validation SM_CHANNEL_TEST=data/test python tf_bert_reviews.py --use-xla=False --use-amp=False --train-batch-size=8 --validation-batch-size=8 --test-batch-size=8 --epochs=2 --train-steps-per-epoch=10 --validation-steps=10 --test-steps=10 --max-seq-length=128 --freeze-bert-layer=False --enable-sagemaker-debugger=True --run-validation=True --run-test=True --run-sample-predictions=True
