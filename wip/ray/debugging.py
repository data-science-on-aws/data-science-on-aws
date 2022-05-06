# https://docs.ray.io/en/latest/ray-observability/ray-debugging.html#getting-started

import ray

ray.init(address="auto")

# On Python 3.6, the breakpoint() function is not supported and you need to use ray.util.pdb.set_trace() instead.

@ray.remote
def f(x):
    try:
        raise Exception
    except:
        breakpoint()
    return x * x

futures = [f.remote(i) for i in range(2)]
print(ray.get(futures))
