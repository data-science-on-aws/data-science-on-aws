import ray

ray.init(address="auto")

df = ray.data.read_csv(paths="data/train/part-algo-1-womens_clothing_ecommerce_reviews.csv")

df.groupby("sentiment").count().show()
