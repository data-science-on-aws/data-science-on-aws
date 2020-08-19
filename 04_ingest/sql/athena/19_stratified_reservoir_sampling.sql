SELECT star_rating,
         review_body,
         product_category
FROM 
    (SELECT *,
         ROW_NUMBER()
        OVER (PARTITION BY star_rating
    ORDER BY  rnd) AS rnk
    FROM 
        (SELECT star_rating,
         review_body,
         product_category,
         RANDOM() AS rnd
        FROM dsoaws.amazon_reviews_parquet ) bucketed ) sampled
    WHERE rnk <= 1000
