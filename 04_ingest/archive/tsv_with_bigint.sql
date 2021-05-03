CREATE TABLE IF NOT EXISTS dsoaws.amazon_reviews_parquet
WITH (format = 'PARQUET', external_location = 's3://sagemaker-us-east-1-806570384721/amazon-reviews-pds/parquet', partitioned_by = ARRAY['product_category']) AS
SELECT marketplace,
         customer_id,
         review_id,
         product_id,
         product_parent,
         product_title,
         star_rating,
         helpful_votes,
         total_votes,
         vine,
         verified_purchase,
         review_headline,
         review_body,
         CAST(YEAR(DATE(review_date)) AS INTEGER) AS year,
         DATE_DIFF('day', DATE('1970-01-01'), DATE(review_date)) AS review_date,
         product_category
FROM dsoaws.amazon_reviews_tsv