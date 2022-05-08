import time
import asyncio
import requests
import starlette
import ray
from ray import serve

ray.init(address="auto")

from ray.experimental.dag.input_node import InputNode

@serve.deployment
async def preprocessor(input_data: str):
    """Simple feature processing that converts str to int"""
    await asyncio.sleep(0.1) # Manual delay for blocking computation
    return int(input_data)

@serve.deployment
async def avg_preprocessor(input_data):
    """Simple feature processing that returns average of input list as float."""
    await asyncio.sleep(0.15) # Manual delay for blocking computation
    return sum(input_data) / len(input_data)

@serve.deployment
class Model:
    def __init__(self, weight: int):
        self.weight = weight

    async def forward(self, input: int):
        await asyncio.sleep(0.3) # Manual delay for blocking computation 
        return f"({self.weight} * {input})"


@serve.deployment
class Combiner:
    def __init__(self, m1: Model, m2: Model):
        self.m1 = m1
        self.m2 = m2

    async def run(self, req_part_1, req_part_2, operation):
        # Merge model input from two preprocessors  
        req = f"({req_part_1} + {req_part_2})"
        
        # Submit to both m1 and m2 with same req data in parallel
        r1_ref = self.m1.forward.remote(req)
        r2_ref = self.m2.forward.remote(req)
        
        # Async gathering of model forward results for same request data
        rst = await asyncio.gather(r1_ref, r2_ref)
        
        # Control flow that determines runtime behavior based on user input
        if operation == "sum":
            return f"sum({rst})"
        else:
            return f"max({rst})"
        
@serve.deployment(num_replicas=2)
class DAGDriver:
    def __init__(self, dag_handle):
        self.dag_handle = dag_handle

    async def predict(self, inp):
        """Perform inference directly without HTTP."""
        return await self.dag_handle.remote(inp)

    async def __call__(self, request: starlette.requests.Request):
        """HTTP endpoint of the DAG."""
        input_data = await request.json()
        return await self.predict(input_data)

# DAG building
with InputNode() as dag_input:
    # Partial access of user input by index
    preprocessed_1 = preprocessor.bind(dag_input[0])
    preprocessed_2 = avg_preprocessor.bind(dag_input[1])
    # Multiple instantiation of the same class with different args
    m1 = Model.bind(1)
    m2 = Model.bind(2)
    # Use other DeploymentNode in bind()
    combiner = Combiner.bind(m1, m2)
    # Use output of function DeploymentNode in bind()
    dag = combiner.run.bind(
        preprocessed_1, preprocessed_2, dag_input[2]
    ) 
    
    # Each serve dag has a driver deployment as ingress that can be user provided.
    serve_dag = DAGDriver.options(route_prefix="/my-dag").bind(dag)

# Note: Trying to change port, but only available through ray.serve.start 
# http_options={"port": 8002}
dag_handle = serve.run(serve_dag)

# Warm up
ray.get(dag_handle.predict.remote(["0", [0, 0], "sum"]))
