pip install -q scikit-learn==0.23.1
pip install -q tensorflow==2.3.0

rm -rf model/
rm -rf output/

#SM_INPUT_DATA_CONFIG={\"train\":{\"TrainingInputMode\":\"Pipe\"}} 

SM_TRAINING_ENV={\"is_master\":true} SAGEMAKER_JOB_NAME=blah-job-name SM_CURRENT_HOST=blah SM_NUM_GPUS=0 SM_HOSTS={\"hosts\":\"blah\"} SM_MODEL_DIR=model/ SM_OUTPUT_DIR=output/ SM_OUTPUT_DATA_DIR=output/data/ SM_CHANNEL_TRAIN=../data-tfrecord/bert-train SM_CHANNEL_VALIDATION=../data-tfrecord/bert-validation SM_CHANNEL_TEST=../data-tfrecord/bert-test python train.py --use_xla=False --use_amp=False --learning_rate=0.00001 --enable_tensorboard=True 
