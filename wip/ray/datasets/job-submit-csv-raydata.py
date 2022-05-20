# DO NOT CHANGE THIS FROM 127.0.0.1... you need to use port-forwarding as described above!!
ray job submit --working-dir . --runtime-env job-raydata-runtime.yaml --address http://127.0.0.1:8265 -- python csv-raydata.py
