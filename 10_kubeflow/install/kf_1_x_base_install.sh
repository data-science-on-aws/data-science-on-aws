echo "source <(kubectl completion bash)" >> ~/.bashrc
source ~/.bashrc

cat <<EOF > /tmp/pv.yml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: volname
spec:
  capacity:
    storage: volsize
  accessModes:
    - ReadWriteOnce
    - ReadOnlyMany
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Recycle
  hostPath:
    path: volpath
EOF

kubectl apply -f https://openebs.github.io/charts/openebs-operator-1.6.0.yaml

kubectl get pods -n openebs

kubectl delete sc openebs-hostpath

cat <<< '
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: openebs-hostpath
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
    openebs.io/cas-type: local
    cas.openebs.io/config: |
      - name: StorageType
        value: "hostpath"
      - name: BasePath
        value: "/root/data/"
provisioner: openebs.io/local
volumeBindingMode: WaitForFirstConsumer
' > openebs-sc.yaml

sleep 10
kubectl create -f openebs-sc.yaml

kubectl get sc

wget https://github.com/kubeflow/kfctl/releases/download/v1.0/kfctl_v1.0-0-g94c35cf_linux.tar.gz

tar -xvf kfctl_v1.0-0-g94c35cf_linux.tar.gz
mv kfctl /usr/local/bin
chmod +x /usr/local/bin/kfctl
rm kfctl_v1.0-0-g94c35cf_linux.tar.gz

#export PATH=$PATH

export KF_NAME=kubeflow-cluster
echo "export KF_NAME=${KF_NAME}" | tee -a ~/.bash_profile

export BASE_DIR=/opt
echo "export BASE_DIR=${BASE_DIR}" | tee -a ~/.bash_profile
export KF_DIR=${BASE_DIR}/${KF_NAME}
echo "export KF_DIR=${KF_DIR}" | tee -a ~/.bash_profile

rm -rf ${KF_DIR}
mkdir -p ${KF_DIR}
cd ${KF_DIR}


export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_k8s_istio.v1.0.0.yaml"
echo "export CONFIG_URI=${CONFIG_URI}" | tee -a ~/.bash_profile

mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl build -V -f ${CONFIG_URI}

export CONFIG_FILE=${KF_DIR}/kfctl_k8s_istio.v1.0.0.yaml
echo "export CONFIG_FILE=${CONFIG_FILE}" | tee -a ~/.bash_profile

cd ${KF_DIR}
rm -rf kustomize
rm -rf .cache

n=0
until [ $n -ge 5 ]
do
   mkdir -p ${KF_DIR}
   cd ${KF_DIR}

   rm -rf kustomize
   rm -rf .cache

   kfctl apply -V -f ${CONFIG_FILE} && break
   n=$[$n+1]
   sleep 2
done

kubectl config set-context \
    $(kubectl config current-context) \
    --namespace kubeflow

kubectl get pod

export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')
echo "export INGRESS_PORT=${INGRESS_PORT}" | tee -a ~/.bash_profile
