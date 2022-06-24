# Note:  You must do "in/job-submit-pytorch-huggingface-clothing.sray attach cluster.yaml -p 8265" to setup the port forward from 127.0.0.1 to the Ray cluster

# DO NOT CHANGE THIS FROM 127.0.0.1... you need to use port-forwarding as described above!!
ray job submit --working-dir . --runtime-env job-pytorch-huggingface-clothing-runtime.yaml --address http://127.0.0.1:8265 -- python pytorch-huggingface-clothing.py --num_train_epochs 1 --max_train_steps 50 --max_length 128 --learning_rate 0.00003 --per_device_train_batch_size 64 --num_workers 2 --model_name_or_path roberta-base --train_file ./data/train/part-algo-1-womens_clothing_ecommerce_reviews.csv --validation_file ./data/validation/part-algo-1-womens_clothing_ecommerce_reviews.csv
