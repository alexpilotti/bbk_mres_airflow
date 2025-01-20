from datetime import datetime, timedelta
import os

from airflow.models.dag import DAG

from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.utils.task_group import TaskGroup

from bbk_mres_airflow import common
from bbk_mres_airflow.token_classification import tasks
from bbk_mres_airflow import utils

VAR_REGION = "region"

TOKEN_PREDICTION_LABELS_RMD = f"token_prediction_labels.Rmd"
TOKEN_PREDICTION_LABELS_OUTPUT_FILENAME = (
    "token_prediction_labels_{predict_region}.html")

TOKEN_PREDICTION_METRICS_RMD = f"token_prediction_metrics.Rmd"
TOKEN_PREDICTION_METRICS_OUTPUT_FILENAME = (
    "token_prediction_metrics_{predict_region}.html")

REGIONS = ["CDR1", "CDR2", "CDR3"]


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
    chain = Variable.get(common.VAR_CHAIN, common.CHAIN_H)
    if chain not in [common.CHAIN_H, common.CHAIN_L, common.CHAIN_HL]:
        raise Exception(f"Invalid chain: {chain}")

    fine_tuning_region = Variable.get(VAR_REGION, None)
    if fine_tuning_region == "FULL":
        fine_tuning_region = None

    git_branch = Variable.get(
        common.VAR_GIT_BBK_MRES_BRANCH,
        common.GIT_BBK_MRES_DEFAULT_BRANCH)

    model_tasks_config = common.load_config()["token_classification"]["models"]

    predict_tasks = []

    predict_regions = [fine_tuning_region]
    if not fine_tuning_region:
        # When fine tuning the whole chain, perform the prediction on the
        # full chain and all CDR regions
        predict_regions += REGIONS

    for model_config in model_tasks_config:
        if not model_config.get("enabled", True):
            continue

        model = model_config["model"]
        model_path = model_config.get("path")
        use_default_model_tokenizer = model_config.get(
            "use_default_model_tokenizer")
        use_accelerate = model_config.get("accelerate", False)
        num_gpus = model_config.get("gpus", tasks.DEFAULT_GPUS)
        batch_size = model_config.get("batch_size", tasks.DEFAULT_BATCH_SIZE)
        model_path_pt = None

        if model_path:
            model_path_pt = os.path.join(
                common.EXTERNAL_MODELS_PATH, model_path)

        task_model_name = model_config.get("task_model_name", model)

        with TaskGroup(group_id=task_model_name) as tg:
            with TaskGroup(group_id=f"training") as tg1:
                task_check_train, task_train = tasks.create_fine_tuning_tasks(
                    model, chain, fine_tuning_region, model_path_pt,
                    use_default_model_tokenizer, task_model_name, num_gpus,
                    batch_size, use_accelerate, git_branch)

            with TaskGroup(group_id=f"predict") as tg1:
                for predict_region in predict_regions:
                    (task_check_predict_ft,
                     task_predict_ft) = tasks.create_label_prediction_tasks(
                        model, chain, fine_tuning_region, predict_region,
                        model_path_pt, use_default_model_tokenizer,
                        task_model_name, False, num_gpus, use_accelerate,
                        git_branch)

                    (task_check_predict_pt,
                     task_predict_pt) = tasks.create_label_prediction_tasks(
                        model, chain, fine_tuning_region, predict_region,
                        model_path_pt, use_default_model_tokenizer,
                        task_model_name, True, num_gpus, use_accelerate,
                        git_branch)

                    task_train >> task_check_predict_ft
                    predict_tasks.extend([task_predict_ft, task_predict_pt])

    with TaskGroup(group_id=f"reports") as tg:
        report_tasks = []

        for predict_region in predict_regions:
            if not predict_region:
                predict_region = "FULL"

            token_prediction_labels_rmd = tasks.create_rmarkdown_task(
                f"token_prediction_labels_rmd_{predict_region}",
                TOKEN_PREDICTION_LABELS_RMD,
                common.OUTPUT_PATH,
                TOKEN_PREDICTION_LABELS_OUTPUT_FILENAME.format(
                    predict_region=predict_region),
                chain, fine_tuning_region, predict_region, git_branch)

            token_prediction_metrics_rmd = tasks.create_rmarkdown_task(
                f"token_prediction_metrics_rmd_{predict_region}",
                TOKEN_PREDICTION_METRICS_RMD,
                common.OUTPUT_PATH,
                TOKEN_PREDICTION_METRICS_OUTPUT_FILENAME.format(
                    predict_region=predict_region),
                chain, fine_tuning_region, predict_region, git_branch)

            predict_tasks >> token_prediction_labels_rmd
            predict_tasks >> token_prediction_metrics_rmd
            report_tasks += [token_prediction_labels_rmd,
                             token_prediction_metrics_rmd]

        data_url = f"{utils.get_base_url()}/data/"

        send_success_email = EmailOperator(
            task_id="send_success_email",
            to="{{ var.value.email_to }}",
            subject="BBK-MRes token classification tasks completed",
            html_content=(
                '<p>'
                'Start time: {{ data_interval_start }}<br/>'
                '</br>'
                'Chain: <b>{{ var.value.chain}}</b></br>'
                'Region: <b>{{ var.value.region}}</b></br>'
                'Branch: <b>{{ var.value.bbk_mres_git_branch}}</b></br>'
                '</p>'
                '<h3>Results</h3>'
                '<p>'
                '<ul>'
                '{% for predict_region in params.predict_regions %}'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                '{{ params.token_prediction_labels_output.format('
                'predict_region=predict_region) }}">'
                'Token prediction labels {{ predict_region }}</a>'
                '  </li>'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                '{{ params.token_prediction_metrics_output.format('
                'predict_region=predict_region) }}">'
                'Token prediction metrics {{ predict_region }}</a>'
                '  </li>'
                '{% endfor %}'
                '</ul>'
                '<br/>'
                '<a href="{{ params.get_dag_run_url(dag.dag_id, run_id) }}">'
                'Mode details</a>'
                '<br/>'
                '<a href="{{ params.data_url }}">All data</a>'
                '</p>'),
            params={
                "data_url": data_url,
                "predict_regions": [r or "FULL" for r in predict_regions],
                "token_prediction_labels_output":
                    TOKEN_PREDICTION_LABELS_OUTPUT_FILENAME,
                "token_prediction_metrics_output":
                    TOKEN_PREDICTION_METRICS_OUTPUT_FILENAME,
                "get_dag_run_url": utils.get_dag_run_url}
            )

        report_tasks >> send_success_email
