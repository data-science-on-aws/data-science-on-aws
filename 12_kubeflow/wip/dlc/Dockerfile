# Use the Deep Learning Container as a base Image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.1.0-cpu-py36-ubuntu18.04

# Add any script or repo as required
#ADD https://github.com/data-science-on-aws/workshop/blob/07ef8914a159012058bf4ad08493dc8da808b57f/12_kubeflow/wip/dlc/code/train_orig.py /opt/ml/code/train.py
ADD ./code/train.py /opt/ml/code/train.py
ADD ./data-tfrecord/ /opt/ml/input/data/

RUN chmod +x /opt/ml/code/train.py

#WORKDIR "/usr/local/bin"
#ENTRYPOINT ["python3", "train_eks.py"]
WORKDIR "/"
CMD ["bin/bash"]

