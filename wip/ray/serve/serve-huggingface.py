from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import subprocess
import time
#from fastapi import Response, status

#ray_cluster = subprocess.Popen(['ray', 'start', '--head', '--include-dashboard', 'false', '--disable-usage-stats'])

#print('Sleeping for 20 seconds...')
#time.sleep(20)

from ray import serve
import ray

ray.init()
#address="127.0.0.1:10001",
#         ignore_reinit_error=True)

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
        # status.HTTP_200_OK
        return "" 


serve.start(detached=True, 
            http_options={"port": 8080})

InvocationsDeployment.deploy()
PingDeployment.deploy()

while True:
    time.sleep(10)
