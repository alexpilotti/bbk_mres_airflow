from airflow.decorators import task_group

from bbk_mres_airflow import common
from bbk_mres_airflow import fs_compare_operators
from bbk_mres_airflow import k8s

VCAB_CSV_PATH = f"{common.DATA_PATH}/final_vcab_with_V_coor.csv"
POPS_DIR = f"{common.DATA_PATH}/pops_result"

FINE_TUNING_INPUT_PATH = f"{common.DATA_PATH}/tokens_data.parquet"
FINE_TUNING_OUTPUT_PATH = (
    f"{common.MODELS_PATH}/" + "token_prediction_{model}_{chain}_{region}/")
FINE_TUNING_OUTPUT_PATH_CHECK = FINE_TUNING_OUTPUT_PATH + "config.json"

PREDICT_METRICS_PATH = (
    f"{common.DATA_PATH}/" +
    "token_predict_metrics_{model}_{chain}_{fine_tuning_region}_" +
    "{predict_region}_{pre_trained}.json")
PREDICT_LABELS_PATH = (
    f"{common.DATA_PATH}/" +
    "token_prediction_{model}_{chain}_{fine_tuning_region}_" +
    "{predict_region}_{pre_trained}.parquet")

COMMON_CMD = (
    "git fetch && git reset --hard origin/{{ params.git_branch }} && "
    "pip3 install -r requirements.txt && "
    "{% if params.accelerate %}"
    "accelerate launch --config_file /data/accelerate.yaml "
    "--num_processes=$(nvidia-smi --list-gpus | wc -l)"
    "{% else %}python{% endif %} "
    "abflow/cli.py ")

FINE_TUNING_CMD = COMMON_CMD + (
    "token-fine-tuning -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }} -b {{ params.batch_size }}"
    "{% if params.region %} --region {{ params.region }}{% endif %}"
    "{% if params.model_path %} -p {{ params.model_path }}{% endif %}"
    "{% if params.frozen_layers %} --frozen-layers "
    "{{ params.frozen_layers }}{% endif %}"
    "{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

PREDICT_CMD = COMMON_CMD + (
    "token-prediction -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output_metrics }} "
    "-c {{ params.chain }} -P {{ params.output_labels}}"
    "{% if params.region %} --region {{ params.region }}{% endif %}"
    "{% if params.model_path %} -p {{ params.model_path }}{% endif %}"
    "{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

PROCESS_VCAB_DATA_CMD = COMMON_CMD + (
    "process-vcab-data -c {{ params.csv_path }} "
    "-p {{ params.pops_dir }} "
    "-o {{ params.tokens_data_path }}")

RMARKDOWN_CMD = (
    "git fetch && git reset --hard origin/{{ params.git_branch }} && "
    "mkdir -p '{{ params.output_base_dir }}/{{ run_id }}' && "
    "/usr/bin/Rscript -e "
    "\"rmarkdown::render('{{ params.rmd_path }}', "
    "output_file = '{{ params.output_base_dir }}/{{ run_id }}/"
    "{{ params.output_filename }}', "
    "params = list({{ params.params }}))\"")

DEFAULT_BATCH_SIZE = 64
DEFAULT_GPUS = 4


def create_fine_tuning_tasks(model, chain, region, model_path=None,
                             use_default_model_tokenizer=None,
                             task_model_name=None, num_gpus=DEFAULT_GPUS,
                             batch_size=DEFAULT_BATCH_SIZE,
                             use_accelerate=False, git_branch="main",
                             frozen_layers=None):
    region_str = region or "FULL"

    input_path = FINE_TUNING_INPUT_PATH
    output_path_check = FINE_TUNING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain, region=region_str)
    output_path = FINE_TUNING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, region=region_str)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_update_model_{task_model_name}_{chain}_{region_str}",
        path1=input_path,
        path2=output_path_check,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_train = k8s.create_pod_operator(
        task_id=f"fine_tuning_{task_model_name}_{chain}_{region_str}",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=FINE_TUNING_CMD,
        params={"model": model,
                "input": input_path,
                "output": output_path,
                "chain": chain,
                "region": region,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "batch_size": batch_size,
                "accelerate": use_accelerate,
                "git_branch": git_branch,
                "frozen_layers": frozen_layers}
    )

    task_check >> task_train

    return task_check, task_train


def create_label_prediction_tasks(model, chain, fine_tuning_region,
                                  predict_region, model_path=None,
                                  use_default_model_tokenizer=None,
                                  task_model_name=None, pre_trained=True,
                                  num_gpus=DEFAULT_GPUS, use_accelerate=False,
                                  git_branch="main"):
    fine_tuning_region_str = fine_tuning_region or "FULL"
    predict_region_str = predict_region or "FULL"
    pre_trained_str = (
        common.PRE_TRAINED if pre_trained else common.FINE_TUNED)

    input_path = FINE_TUNING_INPUT_PATH
    output_metrics_path = PREDICT_METRICS_PATH.format(
        model=task_model_name, chain=chain, predict_region=predict_region_str,
        fine_tuning_region=fine_tuning_region_str, pre_trained=pre_trained_str)
    output_labels_path = PREDICT_LABELS_PATH.format(
        model=task_model_name, chain=chain, predict_region=predict_region_str,
        fine_tuning_region=fine_tuning_region_str, pre_trained=pre_trained_str)

    check_input_paths = [input_path]
    if not pre_trained:
        model_path_check = FINE_TUNING_OUTPUT_PATH_CHECK.format(
            model=task_model_name, chain=chain, region=fine_tuning_region_str)
        check_input_paths.append(model_path_check)

        model_path = FINE_TUNING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain, region=fine_tuning_region_str)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=(f"check_update_predict_{task_model_name}_{chain}_"
                 f"{predict_region_str}_{pre_trained_str}"),
        path1=check_input_paths,
        path2=[output_metrics_path, output_labels_path],
        timeout=0,
        trigger_rule="none_failed"
    )

    task_predict = k8s.create_pod_operator(
        task_id=(f"predict_{task_model_name}_{chain}_"
                 f"{predict_region_str}_{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=PREDICT_CMD,
        params={"model": model,
                "input": input_path,
                "output_metrics": output_metrics_path,
                "output_labels": output_labels_path,
                "chain": chain,
                "region": predict_region,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "accelerate": use_accelerate,
                "git_branch": git_branch}
    )

    task_check >> task_predict

    return task_check, task_predict


@task_group(group_id="VCAb")
def create_process_vcab_data_tasks(git_branch):
    csv_path = VCAB_CSV_PATH
    pops_dir = POPS_DIR
    tokens_data_path = FINE_TUNING_INPUT_PATH

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_process_vcab_data",
        path1=[csv_path, pops_dir],
        path2=tokens_data_path,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_process_vcab_data = k8s.create_pod_operator(
        task_id=f"process_vcab_data",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=PROCESS_VCAB_DATA_CMD,
        params={"csv_path": csv_path,
                "pops_dir": pops_dir,
                "tokens_data_path": tokens_data_path,
                "accelerate": False,
                "git_branch": git_branch}
    )

    task_check >> task_process_vcab_data

    return task_check, task_process_vcab_data


def create_rmarkdown_task(task_id, rmd_path, output_base_dir,
                          output_filename, git_branch="main", **kwargs):
    params = ""
    for param, value in kwargs.items():
        if params:
            params += ", "
        if isinstance(value, list):
            value = ','.join(value)
        if isinstance(value, str):
            value = f"'{value}'"
        elif isinstance(value, bool):
            value = str(value).upper()
        params += f"{param}={value}"

    return k8s.create_pod_operator(
        task_id=task_id,
        image=common.R_CONTAINER_IMAGE,
        command=RMARKDOWN_CMD,
        params={"rmd_path": rmd_path,
                "output_base_dir": output_base_dir,
                "output_filename": output_filename,
                "params": params,
                "git_branch": git_branch},
        trigger_rule="none_failed"
    )
