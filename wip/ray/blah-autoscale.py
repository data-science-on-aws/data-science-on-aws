import ray

@ray.remote(num_cpus=8)
def f():
    print('blah')
    return True

ray.init(address="auto")

[f.remote() for _ in range(100)]
