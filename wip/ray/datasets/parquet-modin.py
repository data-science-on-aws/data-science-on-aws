import modin.pandas as pd
import ray

# ray.init(object_store_memory=78643200)
ray.init(address="auto")
df = pd.read_parquet("s3://dsoaws/parquet")

print(df.groupby("product_category").count())
print('blah')
