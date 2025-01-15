from kubernetes import client
from kubernetes.client import models as k8s
from airflow.providers.cncf.kubernetes.operators import pod

K8S_CONN = "k8s_conn"
POOL = "k8s_gpu_pool"
K8S_POD_STARTUP_TIMEOUT_SECONDS = 1800

DATA_PATH = "/data"


def create_pod_operator(task_id, image, command, params, num_gpus=0,
                        trigger_rule="all_success"):
    data_volume = k8s.V1Volume(
        name="data-volume",
        persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
            claim_name="bbk-mres-data-pvc"
        ),
    )

    data_volume_mount = k8s.V1VolumeMount(
        name="data-volume",
        mount_path=DATA_PATH
    )

    resources = None
    pod_template_dict = None

    if num_gpus > 0:
        resources = client.V1ResourceRequirements(
            limits={"nvidia.com/gpu": num_gpus}
            # limits={"nvidia.com/mig-2g.24gb": num_gpus}
        )

        pod_template_dict = {
                "apiVersion": "v1",
                "kind": "Pod",
                "spec": {
                    "runtimeClassName": "nvidia",
                    "containers": [{"name": "cuda"}]
                }
            }

    return pod.KubernetesPodOperator(
        name=task_id,
        task_id=task_id,
        image=image,
        pod_template_dict=pod_template_dict,
        container_resources=resources,
        arguments=[command],
        volumes=[data_volume],
        volume_mounts=[data_volume_mount],
        is_delete_operator_pod=True,
        get_logs=True,
        log_events_on_failure=True,
        do_xcom_push=False,
        kubernetes_conn_id=K8S_CONN,
        pool=POOL,
        params=params,
        startup_timeout_seconds=K8S_POD_STARTUP_TIMEOUT_SECONDS,
        trigger_rule=trigger_rule
    )
