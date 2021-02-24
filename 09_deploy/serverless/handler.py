try:
    import unzip_requirements
except ImportError:
    pass
from model.model import ServerlessModel
import json

model = ServerlessModel('./model', 
                        'sagemaker-us-east-1-835319576252',
                        'serverless-bert/pytorch_model.tar.gz')


def predict_answer(event, context):
    try:
        print(event['body'])
        body = json.loads(event['body'])
        answer = model.predict(body['question'], body['context'])

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'answer': answer})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
