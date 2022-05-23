## Quick Start
Start your Ray cluster from your local laptop:
```
ray up cluster.yaml
```

Clone this repo to run the examples on your local laptop:
```
git clone https://github.com/data-science-on-aws/data-science-on-aws
```

Run the samples in datasets/, train/, workflow/, serve/, etc.

Tear down your Ray cluster from your local laptop:
```
ray down cluster.yaml
```

## Install JupyterLab and MLflow
Attach to the head node of the Ray cluster
```
ray attach cluster.yaml
```

Install Jupyter Lab on the head node of the Ray cluster:
```
pip install jupyterlab
```

Install S3 Browser extension for JupyterLab
```
pip install jupyterlab-s3-browser

jupyter labextension install jupyterlab-s3-browser

jupyter serverextension enable --py jupyterlab_s3_browser
```

Install Scheduler extension for JupyterLab
```
pip install jupyterlab_scheduler

jupyter labextension install jupyterlab_scheduler

jupyter lab build
```

Install MLflow on the head node of the Ray cluster
```
pip install mlflow
```

## Run JupyterLab and MLflow on the head node of the Ray cluster
From your local laptop, Attach to the head node of the Ray cluster
```
ray attach cluster.yaml
```

Run JupyterLab on the head node of the Ray cluster
```
nohup jupyter lab > jupyterlab.out &
```

Run MLflow UI on the head node of the Ray cluster
```
nohup mlflow ui --host 0.0.0.0 --port 5001 > mlflow.out &
```

## Tunnel ports from local laptop to the head node of the Ray cluster
From your local laptop, tunnel port 8888 to the Ray cluster:
```
ray attach cluster.yaml -p 8888
```

From your local laptop, tunnel port 5001 to the Ray cluster:
```
ray attach cluster.yaml -p 5001
```

From your local laptop, start the dashboard and tunnel port 8265 to the Ray cluster:
```
ray dashboard cluster.yaml # This implicitly tunnels port 8265
```

## Navigate to the JupyterLab and MLflow UIs
From your local laptop, run this command to get the JupyterLab url (and `?token=`) 
```
ray exec cluster.yaml "jupyter server list"
```

Navigate your browser to the URL from above to start using JupyterLab:
```
http://127.0.0.1:8888?token=...
```

![image](https://user-images.githubusercontent.com/1438064/169604655-97f32435-681d-4068-b636-ec06ad3abaa1.png)

Navigate your browser to the following URL to start using MLflow:
```
http://127.0.0.1:5001
```

<img width="1793" alt="image" src="https://user-images.githubusercontent.com/1438064/169713719-9047362d-e7b0-4fb7-aed2-185c4ab06145.png">

## References
* Customize your Ray cluster on AWS as shown here:  https://docs.ray.io/en/master/cluster/cloud.html
