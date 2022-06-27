#!/bin/bash

cat /mnt/var/lib/info/instance.json

if grep isMaster /mnt/var/lib/info/instance.json | grep false;
then        
    echo "This is not master node"
    
    ##################
    # WORKER NODE
    ##################

    echo $HOSTNAME  # ip-172-31-31-104...

    #########################################
    sudo yum install -y amazon-linux-extras
    sudo amazon-linux-extras enable python3.8 
    sudo yum install -y python3.8
    #########################################

    python3 --version
    pip3 --version

    # Update notebook env to use 3.7.10
    #sudo /emr/notebook-env/bin/conda install --name base -y python==3.7.10
    #sudo /emr/notebook-env/bin/conda install -y python==3.7.10

    # Install libraries
    #sudo /emr/notebook-env/bin/pip install -U torch transformers pandas datasets accelerate scikit-learn mlflow ray[all]

    export PATH=/home/hadoop/.local/bin:$PATH
    pip3 install -U scikit-learn ray[all]         #torch transformers pandas datasets accelerate scikit-learn mlflow ray[all]

    aws s3 cp s3://dsoaws/emr/leader_hostname.txt .
    RAY_HEAD_IP=$(<leader_hostname.txt)
    echo "$RAY_HEAD_IP"

    sudo mkdir -p /tmp/ray/
    sudo chmod a+rwx -R /tmp/ray/

    ray start --address=$RAY_HEAD_IP:6379 --object-manager-port=8076 --disable-usage-stats

    exit 0
fi

#############
# MASTER NODE
#############

echo $HOSTNAME > leader_hostname.txt
aws s3 cp leader_hostname.txt s3://dsoaws/emr/

#########################################
sudo yum install -y amazon-linux-extras
sudo amazon-linux-extras enable python3.8 
sudo yum install -y python3.8 # seems to install 3.7 instead??
#########################################

python3 --version # seems to install 3.7 instead??
pip3 --version

export PATH=/home/hadoop/.local/bin:$PATH

# Update notebook env to use 3.7.10
sudo /emr/notebook-env/bin/conda install --name base -y python==3.7.10
sudo /emr/notebook-env/bin/conda install -y python==3.7.10

# Install libraries
sudo /emr/notebook-env/bin/pip install -U scikit-learn ray[all]  # torch transformers pandas datasets accelerate scikit-learn mlflow ray[all]

pip3 install -U scikit-learn ray[all] # torch transformers pandas datasets accelerate scikit-learn mlflow ray[all]

sudo mkdir -p /tmp/ray/
sudo chmod a+rwx -R /tmp/ray/

ray start --head --port=6379 --object-manager-port=8076 --disable-usage-stats
