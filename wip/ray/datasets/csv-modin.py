import modin.pandas as pd
import ray

ray.init(address="auto")

df = pd.read_csv("s3://dsoaws/amazon_reviews_us_Digital_Software_v1_00.tsv", 
                 sep='\t',
                 header=0)

print(df)
print('blah')
