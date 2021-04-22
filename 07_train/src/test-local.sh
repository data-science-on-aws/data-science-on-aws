pip install -q scikit-learn==0.23.1
pip install -q tensorflow==2.3.1
pip install -q transformers==3.5.1
pip install -q sagemaker-tensorflow==2.3.0.1.6.1
pip install -q smdebug==1.0.1

rm -rf model/
rm -rf output/
rm -rf checkpoints/

#SM_INPUT_DATA_CONFIG={\"train\":{\"TrainingInputMode\":\"Pipe\"}} 

SM_TRAINING_ENV={\"is_master\":true} SAGEMAKER_JOB_NAME=blah-job-name SM_CURRENT_HOST=blah SM_NUM_GPUS=0 SM_HOSTS={\"hosts\":\"blah\"} SM_MODEL_DIR=model/ SM_OUTPUT_DIR=output/ SM_OUTPUT_DATA_DIR=output/data/ SM_CHANNEL_TRAIN=../data-tfrecord/bert-train SM_CHANNEL_VALIDATION=../data-tfrecord/bert-validation SM_CHANNEL_TEST=../data-tfrecord/bert-test python tf_bert_reviews.py --use_xla=False --use_amp=False --train_batch_size=1 --validation_batch_size=1 --test_batch_size=1 --epochs=3 --learning_rate=0.00001 --epsilon=0.00000001 --max_seq_length=64 --freeze_bert_layer=False --enable_sagemaker_debugger=False --run_validation=True --run_test=True --run_sample_predictions=True --enable_checkpointing=True --checkpoint_base_path=checkpoints/ --enable_tensorboard=True --test_steps=1 --validation_steps=1 --train_steps_per_epoch=1

saved_model_cli show --all --dir model/tensorflow/saved_model/0/ 

#saved_model_cli run --dir 'model/tensorflow/saved_model/0/' --tag_set serve --signature_def serving_default \
#    --input_exprs 'input_ids=np.zeros((1,64));input_mask=np.zeros((1,64))' # ;segment_ids=np.zeros((1,64))'
