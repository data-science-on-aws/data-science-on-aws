#!/bin/sh

# First, if we're on a worker node, just run our pip installs
# They might get overwritten - we'll need to validate later
if grep isMaster /mnt/var/lib/info/instance.json | grep false; then
    sudo python3 -m pip install -U scikit-learn ray[all]

    RAY_HEAD_IP=$(grep "\"masterHost\":" /emr/instance-controller/lib/info/extraInstanceData.json | cut -f2 -d: | cut -f2 -d\")

    sudo mkdir -p /tmp/ray/
    sudo chmod a+rwx -R /tmp/ray/

    # Wait for ray to be available on the leader node in the background
    cat >/tmp/start_ray.sh <<EOF
#!/bin/sh
echo -n "Waiting for Ray leader node..."
while ( ! nc -z -v $RAY_HEAD_IP 6379); do echo -n "."; sleep 5; done
echo -e "\nRay available...starting!"
ray start --address=$RAY_HEAD_IP:6379 --object-manager-port=8076 --disable-usage-stats
EOF

    chmod +x /tmp/start_ray.sh
    nohup /tmp/start_ray.sh &
    exit 0
fi

# Create a script that can execute in the background.
# Otherwise, the bootstrap will wait too long and fail the cluster startup.
cat >/tmp/install_ray.sh <<EOF
#!/bin/sh
# Wait for EMR to finish provisioning
NODEPROVISIONSTATE="waiting"
echo -n "Waiting for EMR to provision..."
while [ ! "\$NODEPROVISIONSTATE" == "SUCCESSFUL" ]; do
    echo -n "."
    sleep 10
    NODEPROVISIONSTATE=\`sed -n '/localInstance [{]/,/[}]/{
    /nodeProvisionCheckinRecord [{]/,/[}]/ {
    /status: / { p }
    /[}]/a
    }
    /[}]/a
    }' /emr/instance-controller/lib/info/job-flow-state.txt | awk ' { print \$2 }'\`
done
    
echo "EMR provisioned! Continuing with installation..."

# Update notebook env to use python 3.7.10 and install libs
sudo /emr/notebook-env/bin/conda install --name base -y python==3.7.10
sudo /emr/notebook-env/bin/conda install -y python==3.7.10
sudo /emr/notebook-env/bin/pip install -U scikit-learn ray[all]  # torch transformers pandas datasets accelerate scikit-learn mlflow ray[all]

sudo pip3 install -U scikit-learn ray[all] # torch transformers pandas datasets accelerate scikit-learn mlflow ray[all]

sudo mkdir -p /tmp/ray/
sudo chmod a+rwx -R /tmp/ray/

ray start --head --port=6379 --object-manager-port=8076 --disable-usage-stats
EOF

# Execute the script in the background
chmod +x /tmp/install_ray.sh
nohup /tmp/install_ray.sh &