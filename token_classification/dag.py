from datetime import datetime, timedelta

from airflow.models.dag import DAG

from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.providers.sftp.hooks.sftp import SFTPHook
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.utils.task_group import TaskGroup

from bbk_mres_airflow import k8s
from bbk_mres_airflow.token_classification import tasks
from bbk_mres_airflow import utils

MODEL_BALM_PAIRED = "BALM-paired"
MODEL_ANTIBERTY = "AntiBERTy"
MODEL_ANTIBERTA2 = "AntiBERTa2"
MODEL_ESM2_650M = "ESM2-650M"
MODEL_ESM2_150M = "ESM2-150M"
MODEL_ESM2_35M = "ESM2-35M"
MODEL_ESM2_3B = "ESM2-3B"
MODEL_ESM2_8M = "ESM2-8M"

MODEL_NAME_FT_ESM2 = "ft-ESM2"

VAR_CHAIN = "chain"
CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"

EXTERNAL_MODELS_PATH = f"{k8s.DATA_PATH}/pre_trained_models"
BALM_MODEL_PATH = f"{
    EXTERNAL_MODELS_PATH}/BALM-paired_LC-coherence_90-5-5-split_122222/"
FT_ESM2_MODEL_PATH = f"{EXTERNAL_MODELS_PATH}/ESM2-650M_paired-fine-tuning/"

OUTPUT_PATH = f"{k8s.DATA_PATH}/output"

TOKEN_PREDICTION_LABELS_RMD = f"token_prediction_labels.Rmd"
TOKEN_PREDICTION_LABELS_OUTPUT_FILENAME = "token_prediction_labels.html"

TOKEN_PREDICTION_METRICS_RMD = f"token_prediction_metrics.Rmd"
TOKEN_PREDICTION_METRICS_OUTPUT_FILENAME = "token_prediction_metrics.html"


with DAG(
    "BBK-MRes-token-classification",
    default_args={
        "depends_on_past": False,
        "email": ["apilot02@student.bbk.ac.uk"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
    max_active_runs=1,
    description="BBK-MRes Token Classification",
    schedule_interval=None,
    # schedule=timedelta(days=1),
    start_date=datetime(2024, 8, 1),
    catchup=False,
    tags=["bbk"],
) as dag:
    chain = Variable.get(VAR_CHAIN, CHAIN_H)
    if chain not in [CHAIN_H, CHAIN_L, CHAIN_HL]:
        raise Exception(f"Invalid chain: {chain}")

    task_info = [
        (MODEL_ANTIBERTY, None, False, None, 1),
        (MODEL_ANTIBERTA2, None, False, None, 1),
        (MODEL_BALM_PAIRED, BALM_MODEL_PATH, False, None, 1),
        (MODEL_ESM2_8M, None, False, None, 1),
        (MODEL_ESM2_35M, None, False, None, 1),
        (MODEL_ESM2_150M, None, False, None, 1),
        (MODEL_ESM2_650M, None, False, None, 2),
        (MODEL_ESM2_650M, FT_ESM2_MODEL_PATH, True, MODEL_NAME_FT_ESM2, 2),
        (MODEL_ESM2_3B, None, False, None, 4)
    ]

    predict_tasks = []

    for (model, model_path_pt, use_default_model_tokenizer,
         task_model_name, num_gpus) in task_info:
        if not task_model_name:
            task_model_name = model

        with TaskGroup(group_id=task_model_name) as tg:
            with TaskGroup(group_id=f"training") as tg1:
                task_check_train, task_train = tasks.create_fine_tuning_tasks(
                    model, chain, model_path_pt, use_default_model_tokenizer,
                    task_model_name, num_gpus)

            with TaskGroup(group_id=f"predict") as tg1:
                (task_check_predict_ft,
                 task_predict_ft) = tasks.create_label_prediction_tasks(
                    model, chain, model_path_pt, use_default_model_tokenizer,
                    task_model_name, pre_trained=False)

                (task_check_predict_pt,
                 task_predict_pt) = tasks.create_label_prediction_tasks(
                    model, chain, model_path_pt, use_default_model_tokenizer,
                    task_model_name, pre_trained=True)

                task_train >> task_check_predict_ft

                predict_tasks.extend([task_predict_ft, task_predict_pt])

    with TaskGroup(group_id=f"reports") as tg:
        token_prediction_labels_rmd = tasks.create_rmarkdown_task(
            "token_prediction_labels_rmd",
            TOKEN_PREDICTION_LABELS_RMD,
            OUTPUT_PATH,
            TOKEN_PREDICTION_LABELS_OUTPUT_FILENAME,
            chain)

        token_prediction_metrics_rmd = tasks.create_rmarkdown_task(
            "token_prediction_metrics_rmd",
            TOKEN_PREDICTION_METRICS_RMD,
            OUTPUT_PATH,
            TOKEN_PREDICTION_METRICS_OUTPUT_FILENAME,
            chain)


        predict_tasks >> token_prediction_labels_rmd
        predict_tasks >> token_prediction_metrics_rmd

        data_url = f"{utils.get_base_url()}/data/"

        send_success_email = EmailOperator(
            task_id="send_success_email",
            to="{{ var.value.email_to }}",
            subject="BBK-MRes token classification tasks completed",
            html_content=(
                '<p>'
                'Start time: {{ data_interval_start }}<br/>'
                '</p>'
                '<h3>Results</h3>'
                '<ul>'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                'token_prediction_labels.html">Token prediction labels</a>'
                '  </li>'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                'token_prediction_metrics.html">Token prediction metrics</a>'
                '  </li>'
                '</ul>'
                '<br/>'
                '<a href="{{ params.get_dag_run_url(dag.dag_id, run_id) }}">'
                'Mode details</a>'
                '<br/>'
                '<a href="{{ params.data_url }}">All data</a>'
                '</p>'),
            params={
                "data_url": data_url,
                "get_dag_run_url": utils.get_dag_run_url}
            )

        [token_prediction_labels_rmd,
         token_prediction_metrics_rmd] >> send_success_email
