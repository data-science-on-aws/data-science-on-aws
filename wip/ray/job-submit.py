# This assumes that you have used "ray attach -p 8265" or run "ray dashboard" to create a port forward to the cluster on port 8265

# CLI equivalent
# ray job submit --working-dir . --runtime-env job-runtime.yaml --address=127.0.0.1:8265 python scikit-distributed.py

from ray.job_submission import JobSubmissionClient
client = JobSubmissionClient("http://127.0.0.1:8265") 
res = client.submit_job( 
    entrypoint="python scikit-distributed.py",
    runtime_env={
        "working_dir": "./",
        "conda": {
             "dependencies": [
                 { 
		    "pip": ['scikit-learn==0.23.0']
		 }
	      ]
        }
    }
) 

print(res) 
