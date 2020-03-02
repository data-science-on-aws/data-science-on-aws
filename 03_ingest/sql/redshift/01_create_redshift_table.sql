CREATE TABLE amazon_reviews(
         marketplace varchar(1024),
         customer_id varchar(1024),
         review_id varchar(1024),
         product_id varchar(1024),
         product_parent varchar(1024),
         product_title varchar(400),
         product_category varchar(1024),
         star_rating int,
         helpful_votes int,
         total_votes int,
         vine varchar(1024),
         verified_purchase varchar(1024),
         review_headline varchar(128),
         review_body varchar(65536),
         review_date varchar(1024)
)
