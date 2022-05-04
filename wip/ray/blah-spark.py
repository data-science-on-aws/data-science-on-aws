import ray
import raydp

ray.init(address="auto")
spark = raydp.init_spark(
  app_name = "example",
  num_executors = 10,
  executor_cores = 64,
  executor_memory = "256GB"
)

df_from_csv = spark.read.option('delimiter', '\t') \
                        .option('header', True) \
                        .csv('s3a://dsoaws/amazon_reviews_us_Digital_Software_v1_00.tsv')

print(df_from_csv)

df_from_csv.groupBy("product_category").count()
