SELECT product_category,
         COUNT(*) AS count_reviews
FROM amazon_reviews_parquet
WHERE LENGTH(review_body) > 20
GROUP BY  product_category
ORDER BY  count_reviews DESC;
