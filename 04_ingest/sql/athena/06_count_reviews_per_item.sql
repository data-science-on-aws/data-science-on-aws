SELECT product_id,
         product_title,
         count(*) AS count_reviews
FROM amazon_reviews_parquet
WHERE product_category='Digital_Video_Download'
        AND LENGTH(review_body) > 20
GROUP BY  product_id, product_title
ORDER BY  count_reviews DESC;
