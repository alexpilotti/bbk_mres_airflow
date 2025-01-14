from bbk_mres_airflow import fs_compare_operators
from bbk_mres_airflow import k8s

PRE_TRAINED = "PT"
FINE_TUNED = "FT"

MODELS_PATH = f"{k8s.DATA_PATH}/models"

FINE_TUNING_INPUT_PATH = f"{k8s.DATA_PATH}/tokens_data.parquet"
FINE_TUNING_OUTPUT_PATH = (
    f"{MODELS_PATH}/" + "token_prediction_{model}_{chain}/")
FINE_TUNING_OUTPUT_PATH_CHECK = FINE_TUNING_OUTPUT_PATH + "config.json"

PREDICT_METRICS_PATH = (
    f"{k8s.DATA_PATH}/" +
    "token_predict_metrics_{model}_{chain}_{pre_trained}.json")
PREDICT_LABELS_PATH = (
    f"{k8s.DATA_PATH}/" +
    "token_prediction_{model}_{chain}_{pre_trained}.parquet")

PREDICT_GPUS = 2

CUDA_CONTAINER_IMAGE = "registry.bbk-mres:5000/bbk-mres-cuda:latest"
R_CONTAINER_IMAGE = "registry.bbk-mres:5000/bbk-mres-r:latest"

FINE_TUNING_CMD = (
    "git fetch && git reset --hard origin/main && "
    "accelerate launch --multi_gpu --mixed_precision fp16 "
    "--num_processes=$(nvidia-smi --list-gpus | wc -l) "
    "attention_comparison/cli.py "
    "token-fine-tuning -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }} -b {{ params.batch_size }} "
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

PREDICT_CMD = (
    "git fetch && git reset --hard origin/main && "
    "accelerate launch --multi_gpu --mixed_precision fp16 "
    "--num_processes=$(nvidia-smi --list-gpus | wc -l) "
    "attention_comparison/cli.py "
    "token-prediction -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output_metrics }} "
    "-c {{ params.chain }} -P {{ params.output_labels}}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

RMARKDOWN_CMD = (
    "git fetch && git reset --hard origin/main && "
    "mkdir -p '{{ params.output_base_dir }}/{{ run_id }}' && "
    "/usr/bin/Rscript -e "
    "\"rmarkdown::render('{{ params.rmd_path }}', "
    "output_file = '{{ params.output_base_dir }}/{{ run_id }}/"
    "{{ params.output_filename }}', "
    "params = list({{ params.params }}))\"")


def create_fine_tuning_tasks(model, chain, model_path=None,
                             use_default_model_tokenizer=None,
                             task_model_name=None, num_gpus=2, batch_size=64):
    input_path = FINE_TUNING_INPUT_PATH
    output_path_check = FINE_TUNING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain)
    output_path = FINE_TUNING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_update_model_{task_model_name}_{chain}",
        path1=input_path,
        path2=output_path_check,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_train = k8s.create_pod_operator(
        task_id=f"fine_tuning_{task_model_name}_{chain}",
        image=CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=FINE_TUNING_CMD,
        params={"model": model,
                "input": input_path,
                "output": output_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "batch_size": batch_size}
    )

    task_check >> task_train

    return task_check, task_train


def create_label_prediction_tasks(model, chain, model_path=None,
                                  use_default_model_tokenizer=None,
                                  task_model_name=None, pre_trained=True):

    pre_trained_str = (PRE_TRAINED if pre_trained else FINE_TUNED)

    input_path = FINE_TUNING_INPUT_PATH
    output_metrics_path = PREDICT_METRICS_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str)
    output_labels_path = PREDICT_LABELS_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str)

    check_input_paths = [input_path]
    if not pre_trained:
        model_path_check = FINE_TUNING_OUTPUT_PATH_CHECK.format(
            model=task_model_name, chain=chain)
        check_input_paths.append(model_path_check)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=(f"check_update_predict_{task_model_name}_{chain}"
                 f"_{pre_trained_str}"),
        path1=check_input_paths,
        path2=output_metrics_path,
        timeout=0,
        trigger_rule="none_failed"
    )

    if not pre_trained:
        model_path = FINE_TUNING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain)

    task_predict = k8s.create_pod_operator(
        task_id=f"predict_{task_model_name}_{chain}_{pre_trained_str}",
        image=CUDA_CONTAINER_IMAGE,
        num_gpus=PREDICT_GPUS,
        command=PREDICT_CMD,
        params={"model": model,
                "input": input_path,
                "output_metrics": output_metrics_path,
                "output_labels": output_labels_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer}
    )

    task_check >> task_predict

    return task_check, task_predict


def create_rmarkdown_task(task_id, rmd_path, output_base_dir,
                          output_filename, chain):
    return k8s.create_pod_operator(
        task_id=task_id,
        image=R_CONTAINER_IMAGE,
        command=RMARKDOWN_CMD,
        params={"rmd_path": rmd_path,
                "output_base_dir": output_base_dir,
                "output_filename": output_filename,
                "params": f"chain='{chain}', data_path='{k8s.DATA_PATH}'"},
        trigger_rule="none_failed"
    )
