#!/bin/bash

############################################################################################
##### LOCAL PATH TO PROJECT
############################################################################################

LOCAL_PROJECT_BUILD_PATH='/home/sagemaker-user/dsoaws-16093790406854542-p-zh1fcyh2gnk3/sagemaker-dsoaws-16093790406854542-p-zh1fcyh2gnk3-modelbuild'

LOCAL_PROJECT_DEPLOY_PATH='/home/sagemaker-user/dsoaws-16093790406854542-p-zh1fcyh2gnk3/sagemaker-dsoaws-16093790406854542-p-zh1fcyh2gnk3-modeldeploy'

############################################################################################
##### WORKSHOP PATH TO PROJECT CODE
############################################################################################

WORKSHOP_PROJECT_BUILD_PATH='/home/sagemaker-user/workshop/10_pipeline/sagemaker_mlops/sagemaker-project-modelbuild'

WORKSHOP_PROJECT_DEPLOY_PATH='/home/sagemaker-user/workshop/10_pipeline/sagemaker_mlops/sagemaker-project-modeldeploy'

############################################################################################
##### COPY LOCAL TO WORKSHOP
############################################################################################

echo "Copy'ing files from local projects folder to workshop repo..."

cp -R $LOCAL_PROJECT_BUILD_PATH/* $WORKSHOP_PROJECT_BUILD_PATH/

cp -R $LOCAL_PROJECT_DEPLOY_PATH/* $WORKSHOP_PROJECT_DEPLOY_PATH/

############################################################################################
echo "Done."
############################################################################################
