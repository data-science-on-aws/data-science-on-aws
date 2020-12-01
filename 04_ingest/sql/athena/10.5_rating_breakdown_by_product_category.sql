SELECT product_category,
         star_rating,
         COUNT(*) AS count_reviews
FROM amazon_reviews_parquet
WHERE LENGTH(review_body) > 20
GROUP BY  product_category, star_rating
ORDER BY  product_category, star_rating ASC, count_reviews DESC