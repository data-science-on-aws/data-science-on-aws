from __future__ import print_function
import boto3
import base64

import sys
import logging
import traceback
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = boto3.client('cloudwatch')

def lambda_handler(event, context):
    output = []
    success = 0
    failure = 0
    for record in event['records']:
        try:
            #logger.info(f'event: {event}')
            payload = base64.b64decode(record['data'])
            datapoint = float(payload)
            # logger.info(f'avg_star_rating: {payload}')

            client.put_metric_data(
                Namespace='kinesis/analytics/AVGStarRating',
                MetricData=[
                    {
                        'MetricName': 'AVGStarRating',
                        'Dimensions': [
                            {
                                'Name': 'Product Category',
                                'Value': 'All'
                             },
                        ],
                        'Value': datapoint,
                        'StorageResolution': 1
                    }
                ]
            )

            output.append({'recordId': record['recordId'], 'result': 'Ok'})
            success += 1
            print(datapoint)
            
        except Exception as exp:
            output.append({'recordId': record['recordId'], 'result': 'DeliveryFailed'})
            failure += 1
            exception_type, exception_value, exception_traceback = sys.exc_info()
            traceback_string = traceback.format_exception(exception_type, exception_value, exception_traceback)
            err_msg = json.dumps({
                "errorType": exception_type.__name__,
                "errorMessage": str(exception_value),
                "stackTrace": traceback_string
            })
            logger.error(err_msg)

    print('Successfully delivered {0} records, failed to deliver {1} records'.format(success, failure))
    return {'records': output}