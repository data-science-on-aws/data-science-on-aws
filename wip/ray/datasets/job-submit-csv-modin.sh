# Note:  You must do "ray attach cluster.yaml -p 8265" to setup the port forward from 127.0.0.1 to the Ray cluster

# DO NOT CHANGE THIS FROM 127.0.0.1... you need to use port-forwarding as described above!!
ray job submit --working-dir . --runtime-env job-modin-runtime.yaml --address http://127.0.0.1:8265 -- python csv-modin.py #--max_length 64 --num_workers 20 --model_name_or_path distilbert-base-uncased --train_file ./data/train/part-algo-1-womens_clothing_ecommerce_reviews.tsv --validation_file ./data/validation/part-algo-1-womens_clothing_ecommerce_reviews.tsv
