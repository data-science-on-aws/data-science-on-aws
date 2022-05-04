import ray

df = ray.data.read_parquet("s3://dsoaws/parquet")

print(df.count())
print('blah')
