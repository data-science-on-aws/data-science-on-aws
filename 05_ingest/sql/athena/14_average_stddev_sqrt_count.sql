SELECT product_category,
         AVG(star_rating) AS avg_star_rating,
         STDDEV(star_rating) AS stddev_star_rating,
         SQRT(COUNT(*)) AS sqrt_count
FROM dsoaws.amazon_reviews_parquet
WHERE LENGTH(review_body) > 20
GROUP BY  product_category
ORDER BY  avg_star_rating DESC
