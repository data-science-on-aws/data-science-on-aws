# Use the Deep Learning Container as a base Image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.1.0-cpu-py36-ubuntu18.04

# Add any script or repo as required

#ADD ./code/train.py ./code/train.py
#ADD ./data-tfrecord/ opt/ml/input/data/
#ADD ./test_data/ opt/ml/input/data/


# Figure out how to install lustre-client-modules-4.14.186-110.268.amzn1.x86_64
# See https://docs.aws.amazon.com/fsx/latest/LustreGuide/install-lustre-client.html
# https://fsx-lustre-client-repo.s3.amazonaws.com/

# RUN apt-get update -y
# RUN apt-get install gpg-agent -y

# RUN wget -O - https://fsx-lustre-client-repo-public-keys.s3.amazonaws.com/fsx-ubuntu-public-key.asc | apt-key add -

# RUN bash -c 'echo "deb https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu bionic main" > /etc/apt/sources.list.d/fsxlustreclientrepo.list && apt-get update'

# #RUN apt install -y lustre-client-modules-$(uname -r)
# # lustre-client-modules-4.14.186-110.268.amzn1.x86_64

# RUN apt install -y lustre-client-modules-$(uname -r)

#RUN chmod +x ./code/train.py

WORKDIR "/"
CMD ["bin/bash"]

