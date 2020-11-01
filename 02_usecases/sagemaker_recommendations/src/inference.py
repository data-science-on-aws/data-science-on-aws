import json
import subprocess
import sys


def input_handler(data, context):
    transformed_instances = []

    for instance in data:
        instance_str = instance.decode('utf-8')        
        transformed_instances.append(instance_str)

    print(transformed_instances)
    
    transformed_data = {"instances": transformed_instances}
    print(transformed_data)

    transformed_data_json = json.dumps(transformed_data)
    print(transformed_data_json)
    
    return transformed_data_json


def output_handler(response, context):
    response_json = response.json()
    print('response_json: {}'.format(response_json))
    
    predicted_classes_str = json.dumps(response_json)

    response_content_type = context.accept_header
    
    return predicted_classes_str, response_content_type

