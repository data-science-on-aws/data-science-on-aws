import time
import ray

ray.init(address="auto")

@ray.remote
class Counter:
    def __init__(self):
        self.label = 'Counter'
        self.count = 0
    def next(self):
        self.count += 1
        return self.count

counter1 = Counter.options(name="Counter1", lifetime="detached").remote()
counter2 = Counter.options(name="Counter2", lifetime="detached").remote()

c1 = ray.get_actor("Counter1")
print(ray.get([c1.next.remote() for _ in range(100)]))

c2 = ray.get_actor("Counter2")
print(ray.get([c2.next.remote() for _ in range(100)]))

