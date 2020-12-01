SELECT product_id,
         product_title,
         AVG(star_rating) AS avg_star_rating
FROM amazon_reviews_parquet
WHERE LENGTH(review_body) > 20
        AND product_category = 'Digital_Video_Download'
GROUP BY  product_id, product_title;
