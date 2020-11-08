ARG REGION=eu-west-1
ARG ARCH=cpu

# SageMaker PyTorch image
FROM 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.0.0-$ARCH-py3

ARG py_version=3

# Validate that arguments are specified
RUN test $py_version || exit 1

# Install python and nginx
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        jq \
        nginx && \
    if [ $py_version -eq 3 ]; \
       then apt-get install -y --no-install-recommends python3.6-dev \
           && ln -s -f /usr/bin/python3.6 /usr/bin/python; \
       else apt-get install -y --no-install-recommends python-dev; fi && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN cd /tmp && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py 'pip<=18.1' && rm get-pip.py

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN python --version

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl wget


RUN pip install --upgrade pip

RUN pip install gunicorn

RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl && \
    pip3 install  --upgrade torchvision

# RUN pip install --upgrade torch torchvision
    
RUN pip --no-cache-dir install \
        flask \
        pathlib \
        gunicorn \
        gevent \
        scipy \
        sklearn \
        pandas \
        Pillow \
        h5py \
        fastprogress

RUN pip install numpy==1.16


RUN pip install pytorch-pretrained-bert
RUN pip install --no-cache-dir fast-bert --upgrade



ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# # Configure the build for our CUDA configuration.
# ENV CI_BUILD_PYTHON python
# ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# ENV TF_NEED_CUDA 1
# ENV TF_CUDA_COMPUTE_CAPABILITIES=3.7,6.1
# ENV TF_CUDA_VERSION=9.0
# ENV TF_CUDNN_VERSION=7
# ENV CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

ENV PATH="/opt/ml/code:${PATH}"
COPY /bert /opt/ml/code
WORKDIR /opt/ml/code

ENTRYPOINT ["python3", "-u", "/opt/ml/code/serve"]