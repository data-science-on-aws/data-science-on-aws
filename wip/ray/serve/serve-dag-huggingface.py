import time
import asyncio
import requests
import starlette
import ray
from ray import serve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

ray.init()

from ray.experimental.dag.input_node import InputNode

@serve.deployment
class Model:
    def __init__(self, model_version_path: str):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_version_path)
        self.classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    async def forward(self, text: str):
        prediction = self.classifier(text)[0]
        return prediction


@serve.deployment
class Combiner:
    def __init__(self, m1: Model, m2: Model):
        self.m1 = m1
        self.m2 = m2

    async def run(self, text):
        print('run(text): {}'.format(text))

        # Submit to both m1 and m2 with same req data in parallel
        r1_ref = self.m1.forward.remote(text) 
        r2_ref = self.m2.forward.remote(text) 
        
        # Async gathering of model forward results for same request data
        prediction_list = await asyncio.gather(r1_ref, r2_ref)

        # Calculate average of all model predictions
        prediction_avg = int(sum(int(prediction['label']) for prediction in prediction_list) / len(prediction_list))
        return prediction_avg
 

@serve.deployment(num_replicas=1)
class DAGDriver:
    def __init__(self, dag_handle):
        self.dag_handle = dag_handle

    async def predict(self, text):
        """Perform inference directly without HTTP."""
        return await self.dag_handle.remote(text)

    async def __call__(self, request: starlette.requests.Request):
        """HTTP endpoint of the DAG."""
        input_data = await request.body()
        input_data_str = input_data.decode('utf-8')
        return await self.predict(input_data_str)

# DAG building
with InputNode() as dag_input:
    # Multiple instantiation of the same class with different args
    version1_model_path = "./transformer1/" 
    version2_model_path = "./transformer2/" 

    m1 = Model.bind(version1_model_path)
    m2 = Model.bind(version2_model_path)

    # Combine the two models
    combiner = Combiner.bind(m1, m2)


    # Use output of function DeploymentNode in bind()
    dag = combiner.run.bind(
        dag_input
    ) 
    
    # Each serve dag has a driver deployment as ingress that can be user provided.
    serve_dag = DAGDriver.options(route_prefix="/invocations").bind(dag)

dag_handle = serve.run(serve_dag)

# Sample prediction 
text = "Ray Serve is great!"

prediction = ray.get(dag_handle.predict.remote(text))

print("**** Avg prediction for '{}' from '{}' and '{}' is {} ****" \
      .format(text, version1_model_path, version2_model_path, prediction))

while True:
    time.sleep(30)
