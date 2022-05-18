import ray
from ray import workflow

ray.init(
#         address="auto", 
         storage="/tmp"
        )
workflow.init()

# RuntimeError: Unable to initialize storage: /tmp/_valid file created during init not found. Check that configured cluster storage path is readable from all worker nodes of the cluster.

@workflow.step
def one() -> int:
    return 1

@workflow.step
def add(a: int, b: int) -> int:
    return a + b

my_workflow: "Workflow[int]" = add.step(100, one.step())
print(my_workflow)

output = my_workflow.run(workflow_id="run_1")
print(output)

print(workflow.get_status("run_1"))
print(workflow.get_output("run_1"))
