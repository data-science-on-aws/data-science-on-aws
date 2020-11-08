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