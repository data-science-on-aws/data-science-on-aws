import ray

ray.init(address="auto")

df = ray.data.read_parquet("s3://dsoaws/parquet")

print(df.groupby("product_category").count())
print('blah')
