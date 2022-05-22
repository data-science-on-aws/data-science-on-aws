import time
import asyncio
import requests
import starlette
import ray
from ray import serve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from ray.experimental.dag.input_node import InputNode

ray.init() #address="auto")

@serve.deployment
class Model:
    def __init__(self, model_version_path: str):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_version_path)
        self.model_version_path = model_version_path
        self.classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    async def predict(self, text: str):
        prediction = self.classifier(text)[0]
        return prediction


@serve.deployment
class Combiner:
    def __init__(self, m1: Model, m2: Model):
        self.m1 = m1
        self.m2 = m2

    async def run(self, text):
        # Submit to both m1 and m2 with same req data in parallel
        r1_ref = self.m1.predict.remote(text) 
        r2_ref = self.m2.predict.remote(text) 
        
        # Async gathering of model predict results for same request data
        prediction_list = await asyncio.gather(r1_ref, r2_ref)

        # Calculate average of all model predictions
        prediction_avg = int(sum(int(prediction['label']) for prediction in prediction_list) / len(prediction_list))
        return prediction_avg
 

@serve.deployment(num_replicas=1,
                  route_prefix="/invocations")
class DAGDriver:
    def __init__(self, dag):
        self.dag = dag

    async def predict(self, text):
        """Perform inference directly without HTTP."""
        return await self.dag.remote(text)

    async def __call__(self, request: starlette.requests.Request):
        """HTTP endpoint of the DAG."""
        input_data = await request.body()
        input_data_str = input_data.decode('utf-8')
        return await self.predict(input_data_str)


# Setup the dag building
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
    serve_dag = DAGDriver.options(
                ).bind(dag)

dag = serve.run(serve_dag)


# IGNORE THIS - IT'S JUST A LOCAL TEST
# Sample prediction 
text = "Ray Serve is great!"

prediction = ray.get(dag.predict.remote(text))

print("**** Avg prediction for '{}' from '{}' and '{}' is {} ****" \
      .format(text, version1_model_path, version2_model_path, prediction))

while True:
    time.sleep(30)
