# This is a sample Python program that trains a simple PyTorch CIFAR-10 model.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas matplotlib torch torchvision
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
##############################################################################################

import os
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONLinesSerializer
from sagemaker.deserializers import JSONLinesDeserializer

def do_inference_on_local_endpoint(predictor):
    print('Starting Inference on local mode endpoint')

    inputs = [
        {"features": ["I love this product!"]},
        {"features": ["OK, but not great."]},
        {"features": ["This is not the right product."]},
    ]

    predicted_classes = predictor.predict(inputs)

    for predicted_class in predicted_classes:
        print("Predicted class {} with probability {}".format(predicted_class['predicted_label'], predicted_class['probability']))

def main():

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

    print('Deploying local mode endpoint')
    
    model = Model(
        image_uri="dsoaws/ray-serve-sagemaker:1.0",
        model_data="s3://dsoaws/model.tar.gz",
        role=role,
        name="ray-serve-model"
    )

    predictor = model.deploy(
        instance_type='local',
        initial_instance_count=1,
        serializer=JSONLinesSerializer(),
        deserializer=JSONLinesDeserializer(),
        wait=False
    )

    do_inference_on_local_endpoint(predictor)

    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()
