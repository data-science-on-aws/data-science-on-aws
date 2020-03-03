CREATE TABLE amazon_reviews_parquet(
         marketplace varchar(2),
         customer_id varchar(8),
         review_id varchar(14),
         product_id varchar(10),
         product_parent varchar(9),
         product_title varchar(400),
         product_category varchar(24),
         star_rating int,
         helpful_votes int,
         total_votes int,
         vine varchar(1),
         verified_purchase varchar(1),
         review_headline varchar(128),
         review_body varchar(65536),
         review_date varchar(10),
         year int
)