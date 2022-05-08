import ray
import raydp

ray.init(address="auto")
spark = raydp.init_spark(
  app_name = "example",
  num_executors = 2,
  executor_cores = 8,
  executor_memory = "16GB"
)

df_from_csv = spark.read.option('delimiter', '\t') \
                        .option('header', True) \
                        .csv('s3://dsoaws/amazon_reviews_us_Digital_Software_v1_00.tar.gz')

print(df_from_csv)

df_from_csv.groupBy("product_category").count()
