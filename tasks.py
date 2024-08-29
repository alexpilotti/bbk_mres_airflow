import os

import jinja2


from airflow.decorators import task_group
from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.providers.ssh.operators.ssh import SSHOperator

from bbk_mres_airflow import common
from bbk_mres_airflow import sftp_compare_operators


MODELS_PATH = f"{common.DATA_PATH}/models"
VENV_PATH = f"{common.BASE_PATH}/venv"

SPLIT_DATA_INPUT_PATH = f"{common.DATA_PATH}/S_FULL.parquet"

TRAINING_INPUT_PATH = f"{common.DATA_PATH}/S_split.parquet"
TRAINING_OUTPUT_PATH = f"{MODELS_PATH}/" + "{model}_{chain}/"
TRAINING_OUTPUT_PATH_CHECK = TRAINING_OUTPUT_PATH + "config.json"

PRE_TRAINED = "PT"
FINE_TUNED = "FT"
ATTENTIONS_INPUT_PATH = f"{common.DATA_PATH}/S_attentions.parquet"
ATTENTIONS_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "attention_weights_{model}_{chain}_{pre_trained}.parquet")

EMBEDDINGS_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "embeddings_{model}_{chain}_{pre_trained}.pt")

SVM_EMBEDDINGS_PREDICTION_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" + "svm_{model}_{chain}_{pre_trained}.csv")

UCL_SGE_UTILS_BASE_DIR = "/SAN/fraternalilab/bcells/apilotti/sge-utils"

UCL_BASE_DIR = "/SAN/fraternalilab/bcells/apilotti/bbk-mres"
UCL_DATA_PATH = f"{UCL_BASE_DIR}/data"
UCL_MODELS_PATH = f"{UCL_BASE_DIR}/models"
UCL_TRAINING_INPUT_PATH = f"{UCL_DATA_PATH}/S_split.parquet"
UCL_TRAINING_OUTPUT_PATH = f"{UCL_MODELS_PATH}/" + "{model}_{chain}/"

POSITIVE_LABELS = "S+ S1+ S2+"
FOLD = 1

SPLIT_DATA_CMD = (
    "source {{ params.venv_path }}/bin/activate && "
    "python3 {{ params.base_path }}/attention_comparison/cli.py "
    "split-data -i {{ params.input }} -o {{ params.output }} "
    "-l {{ params.positive_labels }} -f {{ params.fold }}")

TRAINING_CMD = (
    "source {{ params.venv_path }}/bin/activate && "
    "python3 {{ params.base_path }}/attention_comparison/cli.py "
    "fine-tuning -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

ATTENTIONS_CMD = (
    "source {{ params.venv_path }}/bin/activate && python3 "
    "{{ params.base_path }}/attention_comparison/cli.py "
    "attentions -m {{ params.model }} -i {{ params.input }} "
    "-o {{ params.output }} -c {{ params.chain }}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

EMBEDDINGS_CMD = (
    "source {{ params.venv_path }}/bin/activate && "
    "python3 {{ params.base_path }}/attention_comparison/cli.py "
    "embeddings -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

SVM_EMBEDDINGS_PREDICTION_CMD = (
    "source {{ params.venv_path }}/bin/activate && "
    "python3 {{ params.base_path }}/attention_comparison/cli.py "
    "svm-embeddings-prediction -i {{ params.input }} "
    "-e {{ params.embeddings }} -o {{ params.output }} "
    "-l {{ params.positive_labels }}")

RMARKDOWN_CMD = (
    "tmp_dir=$(/usr/bin/mktemp -d) && "
    "tmp_rmd_path=\"${tmp_dir}/$(basename '{{ params.rmd_path }}')\" && "
    "cp '{{ params.rmd_path }}' \"${tmp_rmd_path}\" && "
    "/usr/bin/Rscript -e "
    "\"rmarkdown::render('${tmp_rmd_path}', "
    "output_file = '{{ params.output_path }}', "
    "params = list({{ params.params }}))\" && "
    "rm -rf \"${tmp_dir}\"")

'''
QSUB_CMD = (
    "qsub -l tmem=4G -l h_rt=19:00:00 -l gpu={{ params.require_gpu }} -R y "
    "-pe gpu {{ params.num_gpu }} -l tscratch={{ params.stratch_gb }}G -j y "
    "-N {{ params.job_name }} {{ params.cmd }}")
'''

SGE_CMD = (
    f"{UCL_SGE_UTILS_BASE_DIR}/"
    "run_sge_task.sh {{ params.job_name }} {{ params.cmd }}")

UCL_TRAINING_CMD = (
    f"{UCL_BASE_DIR}/" + "fine_tuning.sh {{ model }} "
    "{{ chain }} {{ input }} {{ output }} {{ model_path or '' }} "
    "{{ use_default_model_tokenizer or '' }}")


def create_attention_comparison_tasks(
        ssh_hook, sftp_hook, model, chain, model_path=None,
        use_default_model_tokenizer=None,
        task_model_name=None, pre_trained=True):

    pre_trained_str = (PRE_TRAINED if pre_trained else FINE_TUNED)
    input_path = ATTENTIONS_INPUT_PATH
    output_path = ATTENTIONS_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str)

    if not pre_trained:
        model_path = TRAINING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain)

    task_check = sftp_compare_operators.SFTPComparePathDatetimesSensor(
        task_id=("check_updated_attentions_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        sftp_hook=sftp_hook,
        path1=input_path,
        path2=output_path,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_attention_comparison = SSHOperator(
        task_id=("attention_comparison_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        ssh_hook=ssh_hook,
        command=ATTENTIONS_CMD,
        params={"venv_path": VENV_PATH,
                "base_path": common.BASE_PATH,
                "model": model,
                "input": input_path,
                "output": output_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer},
        pool="single_gpu_pool"
    )

    task_check >> task_attention_comparison

    return task_check, task_attention_comparison


@task_group(group_id="split_data")
def create_split_data_tasks(ssh_hook, sftp_hook):

    input_path = SPLIT_DATA_INPUT_PATH
    # TODO: add chain to path
    output_path = TRAINING_INPUT_PATH

    task_check = sftp_compare_operators.SFTPComparePathDatetimesSensor(
        task_id=f"check_split_data",
        sftp_hook=sftp_hook,
        path1=input_path,
        path2=output_path,
        timeout=0
    )

    task_split_data = SSHOperator(
        task_id=f"split_data",
        ssh_hook=ssh_hook,
        command=SPLIT_DATA_CMD,
        params={"venv_path": VENV_PATH,
                "base_path": common.BASE_PATH,
                "input": input_path,
                "output": output_path,
                "positive_labels": POSITIVE_LABELS,
                "fold": FOLD},
    )

    task_check >> task_split_data

    return task_check, task_split_data


def create_training_tasks(ssh_hook, sftp_hook, model, chain,
                          model_path=None, use_default_model_tokenizer=None,
                          task_model_name=None):

    input_path = TRAINING_INPUT_PATH
    output_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain)
    output_path = TRAINING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain)

    task_check = sftp_compare_operators.SFTPComparePathDatetimesSensor(
        task_id=f"check_update_model_{task_model_name}_{chain}",
        sftp_hook=sftp_hook,
        path1=input_path,
        path2=output_path_check,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_train = SSHOperator(
        task_id=f"training_{task_model_name}_{chain}",
        ssh_hook=ssh_hook,
        command=TRAINING_CMD,
        params={"venv_path": VENV_PATH,
                "base_path": common.BASE_PATH,
                "model": model,
                "input": input_path,
                "output": output_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer},
        pool="single_gpu_pool"
    )

    task_check >> task_train

    return task_check, task_train


def create_embeddings_tasks(ssh_hook, sftp_hook, model, chain,
                            model_path=None, use_default_model_tokenizer=None,
                            task_model_name=None, pre_trained=True):

    pre_trained_str = (PRE_TRAINED if pre_trained else FINE_TUNED)

    input_path = SPLIT_DATA_INPUT_PATH
    embeddings_path = EMBEDDINGS_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str)

    if not pre_trained:
        model_path = TRAINING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain)

    task_check_emb = sftp_compare_operators.SFTPComparePathDatetimesSensor(
        task_id=("check_update_embeddings_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        sftp_hook=sftp_hook,
        path1=input_path,
        path2=embeddings_path,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_embeddings = SSHOperator(
        task_id=f"get_embeddings_{task_model_name}_{chain}_{pre_trained_str}",
        ssh_hook=ssh_hook,
        command=EMBEDDINGS_CMD,
        params={"venv_path": VENV_PATH,
                "base_path": common.BASE_PATH,
                "model": model,
                "input": input_path,
                "output": embeddings_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer},
        pool="single_gpu_pool"
    )

    svm_output_path = SVM_EMBEDDINGS_PREDICTION_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str)

    task_check_svm = sftp_compare_operators.SFTPComparePathDatetimesSensor(
        task_id=("check_update_svm_embeddings_prediction_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        sftp_hook=sftp_hook,
        path1=embeddings_path,
        path2=svm_output_path,
        timeout=0,
        trigger_rule="none_failed"
    )

    task_compute_svm_embeddings_prediction = SSHOperator(
        task_id=("compute_svm_embeddings_prediction_" +
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        ssh_hook=ssh_hook,
        command=SVM_EMBEDDINGS_PREDICTION_CMD,
        params={"venv_path": VENV_PATH,
                "base_path": common.BASE_PATH,
                "input": input_path,
                "output": svm_output_path,
                "embeddings": embeddings_path,
                "positive_labels": POSITIVE_LABELS}
    )

    task_check_emb >> task_embeddings >> task_check_svm
    task_check_svm >> task_compute_svm_embeddings_prediction

    return (task_check_emb, task_embeddings, task_check_svm,
            task_compute_svm_embeddings_prediction)


def create_rmarkdown_task(ssh_hook, task_id, rmd_path, output_path, chain):
    return SSHOperator(
        task_id=task_id,
        ssh_hook=ssh_hook,
        command=RMARKDOWN_CMD,
        params={"rmd_path": rmd_path,
                "output_path": output_path,
                "params": f"chain='{chain}', data_path='{common.DATA_PATH}'"},
        trigger_rule="none_failed"
    )


def _create_grid_engine_task(
        ssh_hook, task_id, cmd, num_gpu=2, stratch_gb=50,
        trigger_rule="all_success", retries=5):
    return SSHOperator(
        task_id=task_id,
        ssh_hook=ssh_hook,
        command=SGE_CMD,
        trigger_rule=trigger_rule,
        retries=retries,
        params={# "require_gpu": num_gpu > 0,
                # "num_gpu": num_gpu,
                # "stratch_gb": stratch_gb,
                "job_name": task_id,
                "cmd": cmd}
    )


@task_group(group_id="ucl_upload_sequences")
def create_ucl_upload_sequences_task(sftp_hook, ucl_sftp_hook):
    tmp_input_path = f"/tmp/{os.path.basename(TRAINING_INPUT_PATH)}"

    task_get = SFTPOperator(
        task_id=f"download_tmp_input_sequences",
        sftp_hook=sftp_hook,
        local_filepath=tmp_input_path,
        remote_filepath=TRAINING_INPUT_PATH,
        operation="get",
        create_intermediate_dirs=False,
        trigger_rule="one_success")

    task_put = SFTPOperator(
        task_id=f"ucl_upload_input_sequences",
        sftp_hook=ucl_sftp_hook,
        local_filepath=tmp_input_path,
        remote_filepath=UCL_TRAINING_INPUT_PATH,
        operation="put",
        create_intermediate_dirs=False)

    task_get >> task_put

    return (task_get, task_put)


def create_ucl_training_tasks(
        ucl_ssh_hook, ucl_sftp_hook, ssh_hook, sftp_hook, model, chain,
        model_path=None, use_default_model_tokenizer=None,
        task_model_name=None, num_gpu=2, stratch_gb=50):

    input_path = TRAINING_INPUT_PATH
    output_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain)
    output_path = TRAINING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain)

    ucl_input_path = UCL_TRAINING_INPUT_PATH
    ucl_output_path = UCL_TRAINING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain)

    ucl_zip_path = f"{os.path.normpath(ucl_output_path)}.zip"
    tmp_zip_path = f"/tmp/{os.path.basename(ucl_zip_path)}"
    zip_path = f"{os.path.normpath(output_path)}.zip"

    task_check = sftp_compare_operators.SFTPComparePathDatetimesSensor(
        task_id=f"check_update_model_{task_model_name}_{chain}",
        sftp_hook=sftp_hook,
        path1=input_path,
        path2=output_path_check,
        timeout=0
    )

    template = jinja2.Environment().from_string(UCL_TRAINING_CMD)
    cmd = template.render(
        model=model, chain=chain, input=ucl_input_path,
        output=ucl_output_path, model_path=model_path,
        use_default_model_tokenizer=use_default_model_tokenizer)

    training_task_id = f"ucl_training_{task_model_name}_{chain}"
    training_task = _create_grid_engine_task(
        ucl_ssh_hook, training_task_id, cmd, num_gpu, stratch_gb)

    task_get_model_zip = SFTPOperator(
        task_id=f"ucl_download_zip_{task_model_name}_{chain}",
        sftp_hook=ucl_sftp_hook,
        local_filepath=tmp_zip_path,
        remote_filepath=ucl_zip_path,
        operation="get",
        create_intermediate_dirs=False)

    task_ucl_delete_model_zip = SSHOperator(
        task_id=f"ucl_delete_zip_{task_model_name}_{chain}",
        ssh_hook=ucl_ssh_hook,
        command="rm {{ params.zip_path }}",
        params={"zip_path": ucl_zip_path})

    task_put_model_zip = SFTPOperator(
        task_id=f"upload_zip_{task_model_name}_{chain}",
        sftp_hook=sftp_hook,
        local_filepath=tmp_zip_path,
        remote_filepath=zip_path,
        operation="put",
        create_intermediate_dirs=False)

    task_unzip_model = SSHOperator(
        task_id=f"unzip_{task_model_name}_{chain}",
        ssh_hook=ssh_hook,
        command=("unzip -o {{ params.zip_path }} -d {{ params.models_path }} "
                 "&& rm {{ params.zip_path }}"),
        params={"zip_path": zip_path,
                "models_path": MODELS_PATH})

    task_check >> training_task >> task_get_model_zip
    task_get_model_zip >> task_ucl_delete_model_zip >> task_put_model_zip
    task_put_model_zip >> task_unzip_model

    return (
        task_check, training_task, task_get_model_zip,
        task_ucl_delete_model_zip, task_put_model_zip,
        task_unzip_model)
