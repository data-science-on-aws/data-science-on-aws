UNLOAD ('SELECT marketplace, customer_id, review_id, product_id, product_parent, product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date FROM amazon_reviews')
TO 's3://sagemaker-us-east-1-806570384721/amazon-reviews-pds/parquet-from-redshift' 
IAM_ROLE 'arn:aws:iam::806570384721:role/Redshift_S3_AmazonReviews'
PARQUET PARALLEL ON 
PARTITION BY (product_category)