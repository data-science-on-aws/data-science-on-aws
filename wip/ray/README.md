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

## Run JupyterLab on the head node of the Ray cluster
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

Run JupyterLab on the head node of the Ray cluster
```

nohup jupyter lab > jupyterlab.out &
```

Back on your local laptop, tunnel port 8888 to the Ray cluster:
```
ray attach cluster.yaml -p 8888
```

Navigate your browser to the following URL to start using JupyterLab:
```
http://127.0.0.1:8888
```

![image](https://user-images.githubusercontent.com/1438064/169604655-97f32435-681d-4068-b636-ec06ad3abaa1.png)

## Instatll MLflow on the head node of the Ray cluster
```
pip install mlflow
```

## Run MLflow UI on the head node of the Ray cluster
```
nohup mlflow ui --host 0.0.0.0 --port 5001 > mlflow.out &
```

## Back on your local laptop, tunnel port 5001 to the Ray cluster:
```
ray attach cluster.yaml -p 5001
```

## References
* Customize your Ray cluster on AWS as shown here:  https://docs.ray.io/en/master/cluster/cloud.html
