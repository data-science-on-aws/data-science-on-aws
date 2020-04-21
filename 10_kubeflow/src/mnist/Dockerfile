#This container contains your model and any helper scripts specific to your model.
# When building the image inside mnist.ipynb the base docker image will be overwritten
FROM tensorflow/tensorflow:1.15.2-py3

ADD model.py /opt/model.py
RUN chmod +x /opt/model.py

ENTRYPOINT ["/usr/local/bin/python"]
CMD ["/opt/model.py"]
