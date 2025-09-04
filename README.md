# AbFlow: Airflow DAGs

This repository contains the DAGs for the AbFlow pipelines:

- Antigen-binding classification
- Paratope prediction

## Installation

The following sections provide instructions on how to get all required
components installed.

## Kubernetes

Any Kubernetes instance will do, ideally with nodes having access to NVIDIA
GPUs with CUDA support, either on-prem or on a public cloud.
The following steps are related to the installation of a new local Kubernets
deployment.

### Install k3s

K3s is a popular way for getting a single or multi-node Kuberenets instance up
and running quickly. Please follow the
[k3s quickstart guide](https://docs.k3s.io/quick-start) and related
documentation.

### Kubernetes NVIDIA configuration

Ensure that your GPU instances are properly configured on your Kubernetes
Linux node(s). The output of the following command should include a list of the
NVIDIA GPUs available on the node and the CUDA version.

```
nvidia-smi
```

Get Helm if not yet installed:

```console
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

Configure the Helm *nvdp* repository:

```console
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
```

Save the following YAML content as *time_slicing_values.yaml*, this example
will instruct Kubernetes to create 10 shares per NVIDIA GPU using
*time slicing*. Alternatively, different strategies such as *MPS*, *MIG* or
*GPU passthrough* can be used. Please see the
[device plugin documentation](https://github.com/NVIDIA/k8s-device-plugin?tab=readme-ov-file#deployment-via-helm)
for additional details.

```yaml
version: v1
sharing:
  timeSlicing:
    renameByDefault: false
    failRequestsGreaterThanOne: false
    resources:
    - name: nvidia.com/gpu
      replicas: 10
```

Install and configure the NVIDIA device plugin in your Kubernetes instance:

```console
helm upgrade -i \
nvidia-device-plugin \
nvdp/nvidia-device-plugin \
--version=0.17.0 \
--namespace nvdp \
--create-namespace \
--set runtimeClassName=nvidia \
--set migStrategy=none \
--set gfd.enabled=true \
--set-file config.map.config=time_slicing_values.yaml
```


## NFS shared storage configuration

If a shared NFS storage is not available, here's an easy way to get a local
instance up and running with Docker:

```console
sudo mkdir -p /data/nfs/bbk-mres
cd ~
mkdir nfs
echo "/exports/data *(rw,fsid=0,sync,no_subtree_check)" >> nfs/exports

docker run \
  --name=nfs-server \
  -d \
  -v /data/nfs/bbk-mres:/exports/data \
  -v $(pwd)/nfs/exports:/etc/exports:ro \
  -e NFS_DISABLE_VERSION_3=1 \
  --privileged \
  -p 2049:2049 \
  --restart=unless-stopped \
  erichough/nfs-server
```

Once the NFS storage is available, to add the corresponding
*PersistentVolume* and *PersistentVolumeClaim* configurations in Kubernetes
save the following YAML content as *bbk-mres-pv-pvc.yaml*, replacing
*SERVER_IP_OR_FQDN* with your NFS server address (can be the local server IP
address in case of a local deployment):

```yaml
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: bbk-mres-data-pv
spec:
  capacity:
    storage: 10Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  nfs:
    server: SERVER_IP_OR_FQDN
    path: "/"
  mountOptions:
    - nfsvers=4
    - acregmax=1
    - acregmin=1
    - acdirmin=1
    - acdirmax=1
  storageClassName: ""
  claimRef:
    namespace: bbk-mres
    name: bbk-mres-data-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bbk-mres-data-pvc
  namespace: bbk-mres
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: ""
  volumeName: bbk-mres-data-pv
```

Next, apply the configuration:

```console
kubectl apply -f bbk-mres-pv-pvc.yaml
```

## Install and configure Airflow

Choose a directory where to store the Airflow configuration, e.g.
*/data/airflow* and create the required directory structure:

```console
sudo mkdir -p /data/airflow
sudo chown $USER.$USER /data/airflow
mkdir -p /data/airflow/{dags,config,logs}
cd /data/airflow
```

Clone this repository in the *dags* subdirectory:
```console
pushd /data/airflow/dags
git clone https://github.com/alexpilotti/bbk_mres_airflow
cd bbk_mres_airflow
cp config.json.template config.json
popd
```

A simple way to get Airflow up and running is by using Docker compose. Start
by getting the default *docker-compose.yaml*:

```console
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.0.6/docker-compose.yaml'
```

Add the following at the end of the file to mount the shared configuration in
all Airflow containers, replacing *SERVER_IP_OR_FQDN* as before:

```yaml
volumes:
  postgres-db-volume:
  data:
    driver_opts:
      type: "nfs"
      o: "addr=SERVER_IP_OR_FQDN,nolock,soft,nfsvers=4,acregmin=1,acregmax=1,acdirmin=1,acdirmax=1"
      device: ":/"
```

Change the following variables:
```yaml
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
AIRFLOW__WEBSERVER__WARN_DEPLOYMENT_EXPOSURE: 'false'
AIRFLOW__WEBSERVER__EXPOSE_CONFIG: 'false'
```

If a SMTP server is used, recommended for receiving pipeline success or failure
emails, add the corresponding environment variables, e.g.:

```yaml
AIRFLOW__SMTP__SMTP_USER: 'your@smtp.user'
AIRFLOW__SMTP__SMTP_PASSWORD: 'your_smtp_password'
AIRFLOW__SMTP__SMTP_MAIL_FROM: 'your@email.from'
AIRFLOW__SMTP__SMTP_HOST: 'your.smpt.host'
AIRFLOW__SMTP__SMTP_PORT: SMTP_PORT
AIRFLOW__SMTP__SMTP_STARTTLS: True
```

In case a reverse proxy is used (e.g. NGINX), recommended if the Airflow web
server is exposed on the Internet, add the following:

```yaml
AIRFLOW__WEBSERVER__BASE_URL: 'https://public_base_address'
```

Once the *docker-compose.yaml* is ready, start Airflow:

```console
docker compose up -d
```

## Docker registry

A local Docker registry can be very useful to share your images with Airflow:

```console
#!/bin/bash
set -e
docker run -d \
-p 5000:5000 \
--name registry \
--restart=unless-stopped \
registry:2.8.3
```

## AbFlow docker images

The Dockerfiles for the Abflow container images used by the pipelines are
available in a
[separate repository](https://github.com/alexpilotti/docker-bbk-mres)
The images need to be built and added to the registry.

## Kubernetes access

Save a copy of your *kubeconfig*, configured as desired, in
*/data/airflow/config/kube_config*. E.g., in the case of a local K3s
deployment:

```console
sudo cp /etc/rancher/k3s/k3s.yaml /data/airflow/config/kube_config
# Note: replace 127.0.0.1 with your server address:
# server: https://YOUR_SERVER_IP:6443
```

## Additional information

Please refer to the following repository for additional information on AbFlow:
- [AbFlow CLI](https://github.com/alexpilotti/bbk-mres)
