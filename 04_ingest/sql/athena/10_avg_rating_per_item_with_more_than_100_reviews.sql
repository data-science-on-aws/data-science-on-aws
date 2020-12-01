SELECT product_id,
         product_title,
         count_reviews,
         avg_star_rating
FROM 
    (SELECT product_id,
         product_title,
         count(*) AS count_reviews,
         AVG(star_rating) AS avg_star_rating
    FROM amazon_reviews_parquet
    WHERE product_category='Digital_Video_Download'
            AND LENGTH(review_body) > 20
    GROUP BY  product_id, product_title)
WHERE count_reviews > 100
ORDER BY  avg_star_rating DESC
