SELECT customer_id,
        review_id,
         product_id,
         product_title,
         review_headline,
         review_body,
         review_date,
         year,
         star_rating,
         sentiment,
         product_category
FROM amazon_reviews_with_sentiment LIMIT 50
