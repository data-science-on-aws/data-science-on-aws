import ray

ray.init(address="auto")

df = ray.data.read_parquet("s3://dsoaws/parquet")

print(df.map_batches(lambda row: row if row["product_category"] == "Books" else None)) #.groupby("product_category").count())
print('blah')
