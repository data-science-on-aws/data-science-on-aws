import ray
import raydp

ray.init(address="auto")

configs={
    "spark.driver.extraJavaOptions": "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
}

spark = raydp.init_spark(
  app_name = "example",
  num_executors = 1,
  executor_cores = 8,
  executor_memory = "1GB",
  configs = configs
)

df_from_csv = spark.read.option('delimiter', ',') \
                        .option('header', True) \
                        .csv('./data/train/part-algo-1-womens_clothing_ecommerce_reviews.csv')

print(df_from_csv)

df_from_csv.groupBy("sentiment").count().show()
