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


RUN pip install --upgrade torch torchvision

    
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
        cudnn \
        fastprogress

RUN pip install numpy==1.16

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

RUN pip install pytorch-pretrained-bert
RUN pip --no-cache-dir install git+https://e791691795db788356f2d576c50aa90829425c7e@github.com/kaushaltrivedi/energy-bert.git --upgrade


ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


ENV PATH="/opt/ml/code:${PATH}"
COPY /bert /opt/ml/code
WORKDIR /opt/ml/code