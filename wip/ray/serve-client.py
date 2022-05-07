# Note: Before running this from your laptop, you must run "ray attach cluster.yaml -p 8000" to setup a port-forward from the laptop's port 8000  to the cluster's internal port 8000

# The other option is to use "ray submit" to run this on the cluster as-is without a port-forward

import requests

input_text = "Ray Serve eases the pain of model serving"

result = requests.get("http://127.0.0.1:8000/sentiment", data=input_text).text

print("Result for '{}': {}".format(input_text, result))
