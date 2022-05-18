# Note:  You must do "ray attach cluster.yaml -p 8265" to setup the port forward from 127.0.0.1 to the Ray cluster

# DO NOT CHANGE THIS FROM 127.0.0.1... you need to use port-forwarding as described above!!
ray job submit --working-dir . --runtime-env job-serve-runtime.yaml --address http://127.0.0.1:8265 -- python serve-huggingface.py 
