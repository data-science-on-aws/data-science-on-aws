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

## Run JupyterLab in the Ray cluster
Attach to the head node of the Ray cluster
```
ray attach cluster.yaml
```

Install and run JupyterLab on the head node of the Ray cluster:
```
pip install jupyterlab

jupyter lab
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

## References
* Customize your Ray cluster on AWS as shown here:  https://docs.ray.io/en/master/cluster/cloud.html
