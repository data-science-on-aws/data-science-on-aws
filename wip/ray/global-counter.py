import time
import ray

ray.init(address="auto")


@ray.remote
class Counter:
   def __init__(self):
      self.count = 0
   def inc(self, n):
      self.count += n
   def get(self):
      return self.count

# on the driver
counter = Counter.options(name="global_counter").remote()
print(ray.get(counter.get.remote()))  # get the latest count

# in your envs
counter = ray.get_actor("global_counter")
for i in range(1000):
    counter.inc.remote(1)  # async call to increment the global count

time.sleep(5)
ray.get(counter.get.remote())
