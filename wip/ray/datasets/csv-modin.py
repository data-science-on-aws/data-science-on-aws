import modin.pandas as pd
import ray

ray.init(address="auto")

df = pd.read_csv("data/train/part-algo-1-womens_clothing_ecommerce_reviews.csv", 
                 sep=',',
                 header=0)

print(df.groupby("sentiment").count())
