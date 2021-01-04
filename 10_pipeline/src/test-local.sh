rm -rf model/
rm -rf output/
#SM_INPUT_DATA_CONFIG={\"train\":{\"TrainingInputMode\":\"Pipe\"}} 

SM_TRAINING_ENV={\"is_master\":true} SAGEMAKER_JOB_NAME=blah-job-name SM_CURRENT_HOST=blah SM_NUM_GPUS=0 SM_HOSTS={\"hosts\":\"blah\"} SM_MODEL_DIR=model/ SM_OUTPUT_DIR=output/ SM_OUTPUT_DATA_DIR=output/data/ SM_CHANNEL_TRAIN=../data-tfrecord/bert-train SM_CHANNEL_VALIDATION=../data-tfrecord/bert-validation SM_CHANNEL_TEST=../data-tfrecord/bert-test python tf_bert_reviews.py --use_xla=False --use_amp=False --train_batch_size=8 --validation_batch_size=8 --test_batch_size=8 --epochs=5 --learning_rate=0.00003 --epsilon=0.00000008 --max_seq_length=128 --freeze_bert_layer=False --enable_sagemaker_debugger=False --run_validation=True --run_test=True --run_sample_predictions=True --enable_checkpointing=True --checkpoint_base_path=checkpoints/ --enable_tensorboard=True --test_steps=10 --validation_steps=10 --train_steps_per_epoch=10
