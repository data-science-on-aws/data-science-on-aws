import modin.pandas as pd
import ray

ray.init(object_store_memory=78643200)

df = pd.read_parquet("s3://dsoaws/parquet")

print(df.count())
print('blah')
