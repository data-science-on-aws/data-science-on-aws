from __future__ import print_function

import base64
import os
import io
import boto3
import json

# grab environment variables
JSON_CONTENT_TYPE = 'application/json'
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
print('Endpoint: {}'.format(ENDPOINT_NAME))
runtime= boto3.client('runtime.sagemaker')

print('Loading function')

def lambda_handler(event, context):
    output = []
    
    r = event['records']
    print('records: {}'.format(r))
    print('type_records: {}'.format(type(r)))
    

    for record in event['records']:
        print(record['recordId'])
        payload = base64.b64decode(record['data'])
        print('payload: {}'.format(payload))
        text = payload.decode("utf-8")
        print('text: {}'.format(text))

        # Do custom processing on the payload here
        split_inputs = text.split("\t")
        print(type(split_inputs))
        print(split_inputs)
        review_body = split_inputs[2]
        print(review_body)
        
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            # ContentType='text/csv',
            Body=review_body.encode('utf-8'))
        print('response: {}'.format(response))
                                       
        result = json.loads(response['Body'].read().decode())
        print('result: {}'.format(result))
        
        # Built output_record
        # review_id, star_rating, product_category, review_body
        output_data = '{}\t{}\t{}\t{}'.format(split_inputs[0], str(result), split_inputs[1], review_body)
        print('output_data: {}'.format(output_data))
        output_data_encoded = output_data.encode('utf-8')

        output_record = {
            'recordId': record['recordId'],
            'result': 'Ok',
            'data': base64.b64encode(output_data_encoded).decode('utf-8')
        }
        output.append(output_record)

    print('Successfully processed {} records.'.format(len(event['records'])))
    print('type(output): {}'.format(type(output)))
    print('Output Length: {} .'.format(len(output)))

    return {'records': output}