import modin.pandas as pd
import ray

# ray.init(object_store_memory=78643200)
ray.init(address="auto")
df = pd.read_csv("data/train/part-algo-1-womens_clothing_ecommerce_reviews.csv")

print(df.groupby("sentiment").count())
