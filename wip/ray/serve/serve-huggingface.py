from transformers import pipeline
from ray import serve
import ray

ray.init(address="auto",
         namespace="huggingface-classifier",
         ignore_reinit_error=True)

@serve.deployment(route_prefix="/sentiment", name="sentiment")
class SentimentDeployment:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")

    async def __call__(self, request):
        data = await request.body()
        [result] = self.classifier(str(data))
        return result["label"]

serve.start(detached=True, 
            http_options={"port": 8001})

SentimentDeployment.deploy()
