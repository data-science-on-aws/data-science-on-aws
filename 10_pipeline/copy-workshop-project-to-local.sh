#!/bin/bash

############################################################################################
##### LOCAL PATH TO PROJECT
############################################################################################

LOCAL_PROJECT_BUILD_PATH='/home/sagemaker-user/dsoaws-16093790406854542-p-zh1fcyh2gnk3/sagemaker-dsoaws-16093790406854542-p-zh1fcyh2gnk3-modelbuild'

LOCAL_PROJECT_DEPLOY_PATH='/home/sagemaker-user/dsoaws-16093790406854542-p-zh1fcyh2gnk3/sagemaker-dsoaws-16093790406854542-p-zh1fcyh2gnk3-modeldeploy'

############################################################################################
##### WORKSHOP PATH TO PROJECT CODE
############################################################################################

WORKSHOP_PROJECT_BUILD_PATH='/home/sagemaker-user/workshop/10_pipeline/project-dsoaws-p-ibxfjw9nuim7/sagemaker-project-dsoaws-p-ibxfjw9nuim7-modelbuild'

WORKSHOP_PROJECT_DEPLOY_PATH='/home/sagemaker-user/workshop/10_pipeline/project-dsoaws-p-ibxfjw9nuim7/sagemaker-project-dsoaws-p-ibxfjw9nuim7-modeldeploy'

############################################################################################
##### COPY WORKSHOP TO LOCAL
############################################################################################

echo "Copy'ing files from workshop repo to local projects folder..."
cp -r $WORKSHOP_PROJECT_BUILD_PATH/. $LOCAL_PROJECT_BUILD_PATH/

#cp $WORKSHOP_PROJECT_BUILD_PATH/codebuild-buildspec.yml $LOCAL_PROJECT_BUILD_PATH/codebuid-buildspec.yml
#cp $WORKSHOP_PROJECT_BUILD_PATH/setup.py $LOCAL_PROJECT_BUILD_PATH/setup.py

#cp $WORKSHOP_PROJECT_BUILD_PATH/99_Create_Sagemaker_Pipeline_BERT_Reviews_MLOps.ipynb $LOCAL_PROJECT_BUILD_PATH/99_Create_Sagemaker_Pipeline_BERT_Reviews_MLOps.ipynb

#cp $WORKSHOP_PROJECT_BUILD_PATH/pipelines/run_pipeline.py $LOCAL_PROJECT_BUILD_PATH/pipelines/run_pipeline.py

#cp $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/evaluate_model_metrics.py $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/evaluate_model_metrics.py

#cp $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/inference.py $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/inference.py

#cp $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/pipeline.py $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/pipeline.py

#cp $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/preprocess-scikit-text-to-bert-feature-store.py $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/preprocess-scikit-text-to-bert-feature-store.py

#cp $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/tf_bert_reviews.py $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/tf_bert_reviews.py

###########################################################################################

cp -r $WORKSHOP_PROJECT_DEPLOY_PATH/. $LOCAL_PROJECT_DEPLOY_PATH/

#cp $WORKSHOP_PROJECT_DEPLOY_PATH/prod-config.json $LOCAL_PROJECT_DEPLOY_PATH/prod-config.json 

#cp $WORKSHOP_PROJECT_DEPLOY_PATH/staging-config.json LOCAL_PROJECT_DEPLOY_PATH/staging-config.json

############################################################################################

echo "Done."

############################################################################################
