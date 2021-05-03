SELECT product_category, MIN(DATE_FORMAT(DATE_ADD('day', review_date, DATE_PARSE('1970-01-01','%Y-%m-%d')), '%Y-%m-%d')) AS first_review_date 
FROM amazon_reviews_parquet 
GROUP BY product_category
ORDER BY first_review_date
