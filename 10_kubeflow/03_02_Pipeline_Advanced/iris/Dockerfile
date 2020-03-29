FROM tensorflow/tensorflow:1.15.2-py3

RUN pip install pandas sklearn kfp boto3 --user

COPY iris_comp.py .

ENTRYPOINT [ "bash" ]
