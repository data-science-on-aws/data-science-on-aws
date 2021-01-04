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
##### COPY LOCAL TO WORKSHOP
############################################################################################

echo "Copy'ing files from local projects folder to workshop repo..."

cp -r $LOCAL_PROJECT_BUILD_PATH/. $WORKSHOP_PROJECT_BUILD_PATH/

#cp $LOCAL_PROJECT_BUILD_PATH/codebuid-buildspec.yml $WORKSHOP_PROJECT_BUILD_PATH/codebuild-buildspec.yml

#cp $LOCAL_PROJECT_BUILD_PATH/setup.py $WORKSHOP_PROJECT_BUILD_PATH/setup.py

#cp $LOCAL_PROJECT_BUILD_PATH/99_Create_Sagemaker_Pipeline_BERT_Reviews_MLOps.ipynb $WORKSHOP_PROJECT_BUILD_PATH/99_Create_Sagemaker_Pipeline_BERT_Reviews_MLOps.ipynb 

#cp $LOCAL_PROJECT_BUILD_PATH/pipelines/run_pipeline.py $WORKSHOP_PROJECT_BUILD_PATH/pipelines/run_pipeline.py 

#cp $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/evaluate_model_metrics.py $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/evaluate_model_metrics.py 

#cp $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/inference.py $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/inference.py 

#cp $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/pipeline.py $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/pipeline.py 
#cp $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/preprocess-scikit-text-to-bert-feature-store.py $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/preprocess-scikit-text-to-bert-feature-store.py 

#cp $LOCAL_PROJECT_BUILD_PATH/pipelines/dsoaws/tf_bert_reviews.py $WORKSHOP_PROJECT_BUILD_PATH/pipelines/dsoaws/tf_bert_reviews.py

###########################################################################################

cp -r $LOCAL_PROJECT_DEPLOY_PATH/. $WORKSHOP_PROJECT_DEPLOY_PATH/

#cp $LOCAL_PROJECT_DEPLOY_PATH/prod-config.json $WORKSHOP_PROJECT_DEPLOY_PATH/prod-config.json

#cp $LOCAL_PROJECT_DEPLOY_PATH/staging-config.json $WORKSHOP_PROJECT_DEPLOY_PATH/staging-config.json

############################################################################################
echo "Done."
############################################################################################
