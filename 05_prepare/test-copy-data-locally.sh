# raw-labeled-split-unbalanced-header
mkdir -p ./feature-store/amazon-reviews/csv/raw-labeled-split-unbalanced-header-train/
mkdir -p ./feature-store/amazon-reviews/csv/raw-labeled-split-unbalanced-header-validation/
mkdir -p ./feature-store/amazon-reviews/csv/raw-labeled-split-unbalanced-header-test/
aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/raw-labeled-split-unbalanced-header-train/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/raw-labeled-split-unbalanced-header-train/
aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/raw-labeled-split-unbalanced-header-validation/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/raw-labeled-split-unbalanced-header-validation/
aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/raw-labeled-split-unbalanced-header-test/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/raw-labeled-split-unbalanced-header-test/

# raw-labeled-split-balanced-header
mkdir -p ./feature-store/amazon-reviews/csv/raw-labeled-split-balanced-header-train/
mkdir -p ./feature-store/amazon-reviews/csv/raw-labeled-split-balanced-header-validation/
mkdir -p ./feature-store/amazon-reviews/csv/raw-labeled-split-balanced-header-test/
aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/raw-labeled-split-balanced-header-train/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/raw-labeled-split-balanced-header-train/
aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/raw-labeled-split-balanced-header-validation/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/raw-labeled-split-balanced-header-validation/
aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/raw-labeled-split-balanced-header-test/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/raw-labeled-split-balanced-header-test/

## NOT USING:  tfidf-labeled-split-unbalanced-header
#mkdir -p ./feature-store/amazon-reviews/csv/tfidf-labeled-split-unbalanced-header-train/
#mkdir -p ./feature-store/amazon-reviews/csv/tfidf-labeled-split-unbalanced-header-validation/
#mkdir -p ./feature-store/amazon-reviews/csv/tfidf-labeled-split-unbalanced-header-test/
#aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/tfidf-labeled-split-unbalanced-header-train/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/tfidf-labeled-split-unbalanced-header-train/
#aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/tfidf-labeled-split-unbalanced-header-validation/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/tfidf-labeled-split-unbalanced-header-validation/
#aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/tfidf-labeled-split-unbalanced-header-test/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/tfidf-labeled-split-unbalanced-header-test/

## GENERATED:  tfidf-labeled-split-balanced-header
#mkdir -p ./feature-store/amazon-reviews/csv/tfidf-labeled-split-balanced-header-train/
#mkdir -p ./feature-store/amazon-reviews/csv/tfidf-labeled-split-balanced-header-validation/
#mkdir -p ./feature-store/amazon-reviews/csv/tfidf-labeled-split-balanced-header-test/
#aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/tfidf-labeled-split-balanced-header-train/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/tfidf-labeled-split-balanced-header-train/
#aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/tfidf-labeled-split-balanced-header-validation/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/tfidf-labeled-split-balanced-header-validation/
#aws s3 cp s3://sagemaker-us-east-1-835319576252/sagemaker-scikit-learn-2020-03-09-21-39-23-619/output/tfidf-labeled-split-balanced-header-test/part-algo-2-amazon_reviews_us_Digital_Software_v1_00.csv ./feature-store/amazon-reviews/csv/tfidf-labeled-split-balanced-header-test/


