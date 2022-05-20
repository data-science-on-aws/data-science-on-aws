# Note: Before running this from your laptop, you must run "ray attach cluster.yaml -p 8000" to setup a port-forward from the laptop's port 8000  to the cluster's internal port 8000

# The other option is to use "ray submit" to run this on the cluster as-is without a port-forward

import requests

input_text_list = ["Ray Serve is great!", "Serving frameworks without DAG support are not great."]

for input_text in input_text_list:
    prediction = requests.get("http://127.0.0.1:8000/invocations", data=input_text).text
    print("Prediction for '{}' is {}".format(input_text, prediction))
