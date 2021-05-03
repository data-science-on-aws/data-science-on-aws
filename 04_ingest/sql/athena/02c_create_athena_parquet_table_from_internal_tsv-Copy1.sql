CREATE TABLE amazon_reviews_parquet_from_tsv
WITH ( format = 'PARQUET', external_location = 's3://sagemaker-us-east-1-806570384721/amazon-reviews-pds/parquet-from-tsv' ) AS
SELECT marketplace,
         customer_id,
         review_id,
         product_id,
         product_parent,
         product_title,
         product_category,
         star_rating,
         helpful_votes,
         total_votes,
         vine,
         verified_purchase,
         review_headline,
         review_body,
         review_date
FROM amazon_reviews_tsv