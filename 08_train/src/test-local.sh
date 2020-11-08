pip install -q scikit-learn==0.20.0
pip install -q tensorflow==2.3.0
pip install -q transformers==3.1.0

rm -rf model/
rm -rf output/
rm -rf checkpoints/

#SM_INPUT_DATA_CONFIG={\"train\":{\"TrainingInputMode\":\"Pipe\"}} 

SM_TRAINING_ENV={\"is_master\":true} SAGEMAKER_JOB_NAME=blah-job-name SM_CURRENT_HOST=blah SM_NUM_GPUS=0 SM_HOSTS={\"hosts\":\"blah\"} SM_MODEL_DIR=model/ SM_OUTPUT_DIR=output/ SM_OUTPUT_DATA_DIR=output/data/ SM_CHANNEL_TRAIN=../data-tfrecord/bert-train SM_CHANNEL_VALIDATION=../data-tfrecord/bert-validation SM_CHANNEL_TEST=../data-tfrecord/bert-test python tf_bert_reviews.py --use_xla=False --use_amp=False --train_batch_size=128 --validation_batch_size=128 --test_batch_size=128 --epochs=3 --learning_rate=0.00001 --epsilon=0.00000001 --max_seq_length=64 --freeze_bert_layer=False --enable_sagemaker_debugger=False --run_validation=True --run_test=True --run_sample_predictions=True --enable_checkpointing=True --checkpoint_base_path=checkpoints/ --enable_tensorboard=True --test_steps=100 --validation_steps=100 --train_steps_per_epoch=100
