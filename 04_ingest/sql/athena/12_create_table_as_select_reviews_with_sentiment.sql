CREATE TABLE IF NOT EXISTS amazon_reviews_with_sentiment 
WITH (
  format = 'PARQUET', 
  external_location = 's3://sagemaker-us-east-1-835319576252/amazon_reviews_with_sentiment/', 
      partitioned_by = ARRAY['product_category'] 
)
AS 
SELECT customer_id,
         review_id,
         product_id,
         product_title,
         review_headline,
         review_body,
         review_date,
         year,
         star_rating,
         CASE
             WHEN star_rating > 3 THEN 1
             ELSE 0
         END AS sentiment,
         product_category
FROM amazon_reviews_parquet