from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from ray import serve
import ray

#from fastapi import Response, status

#ray.init(address="auto",
#         ignore_reinit_error=True)

ray.init()

@serve.deployment(route_prefix="/sentiment", name="sentiment")
class SentimentDeployment:
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

SentimentDeployment.deploy()
