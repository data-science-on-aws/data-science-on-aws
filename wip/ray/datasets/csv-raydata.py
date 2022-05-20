import ray
import os

ray.init(address="auto")

print(os.getcwd())

files = os.listdir('.')
print(files)

df = ray.data.read_csv(paths="data/train/part-algo-1-womens_clothing_ecommerce_reviews.csv")

df.groupby("sentiment").count()
