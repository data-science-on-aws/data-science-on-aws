from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import subprocess
import time
from ray import serve
import ray

#ray.init(include_dashboard=False, ignore_reinit_error=True)


import subprocess
import os
import time
import ray
import socket
import json
import sys

class RayHelper():
    def __init__(self, ray_port:str="9339", redis_pass:str="redis_password"):
        
        self.ray_port = ray_port
        self.redis_pass = redis_pass

    
    def start_ray(self):
    
        output = subprocess.run(['ray', 'start', '--head', '-vvv', '--port', self.ray_port, '--redis-password', self.redis_pass, '--include-dashboard', 'false', '--temp-dir', '/opt/ml/ray'], stdout=subprocess.PIPE)
        print(output.stdout.decode("utf-8"))
        ray.init(address="auto", include_dashboard=False, _temp_dir="/opt/ml/ray")
        self._wait_for_workers()
        print("All workers present and accounted for")
        print(ray.cluster_resources())

    
    def _wait_for_workers(self, timeout=60):
        
        print(f"Waiting {timeout} seconds for 1 node to join")
        
        while len(ray.nodes()) < 1:
            print(f"{len(ray.nodes())} nodes connected to cluster")
            time.sleep(5)
            timeout-=5
            if timeout==0:
                raise Exception("Max timeout for nodes to join exceeded")


print(os.environ)

ray_helper = RayHelper()
ray_helper.start_ray()


@serve.deployment(route_prefix="/invocations", name="invocations")
class InvocationsDeployment:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("./transformer/")
        self.classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    async def __call__(self, request):
        data = await request.body()
        [result] = self.classifier(str(data))
        return result["label"]


@serve.deployment(route_prefix="/ping", name="ping")
class PingDeployment:
    def __init__(self):
        pass

    async def __call__(self, request):
        data = await request.body()
        return "" 


serve.start(detached=True, 
            http_options={"port": 8080})

InvocationsDeployment.deploy()
PingDeployment.deploy()

while True:
    time.sleep(30)
