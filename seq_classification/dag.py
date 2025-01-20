from datetime import datetime, timedelta
import os

from airflow.models.dag import DAG

from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.providers.sftp.hooks.sftp import SFTPHook
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.utils.task_group import TaskGroup

from bbk_mres_airflow import common
from bbk_mres_airflow import git_tasks
from bbk_mres_airflow.seq_classification import tasks
from bbk_mres_airflow import ssh_jump_hook
from bbk_mres_airflow import utils

VAR_CHAIN = "chain"
VAR_GIT_BBK_MRES_BRANCH = "bbk_mres_git_branch"
GIT_BBK_MRES_DEFAULT_BRANCH = "main"
GIT_DEFAULT_SGE_UTILS_BRANCH = "master"

VAR_UCL_EXTERNAL_MODELS_PATH = "ucl_external_models_path"

CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"

ATTENTIONS_RMD = "attention_comparison.Rmd"
ATTENTIONS_RMD_OUTPUT_FILENAME = "attention_comparison.html"

CV_AUROC_RMD = "cv_auroc.Rmd"
CV_AUROC_RMD_OUTPUT_FILENAME = "cv_auroc.html"

CV_METRICS_RMD = "metrics.Rmd"
CV_METRICS_RMD_OUTPUT_FILENAME = "metrics.html"

EXTERNAL_MODELS_PATH = f"{common.DATA_PATH}/pre_trained_models"

DEFAULT_GPUS = 2


with DAG(
    "BBK-MRes-sequence-classification",
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
    description="BBK-MRes Sequence Classification",
    schedule_interval=None,
    # schedule=timedelta(days=1),
    start_date=datetime(2024, 8, 1),
    catchup=False,
    tags=["bbk"],
) as dag:
    chain = Variable.get(VAR_CHAIN, CHAIN_H)
    if chain not in [CHAIN_H, CHAIN_L, CHAIN_HL]:
        raise Exception(f"Invalid chain: {chain}")

    git_branch = Variable.get(
        VAR_GIT_BBK_MRES_BRANCH, default_var=GIT_BBK_MRES_DEFAULT_BRANCH)

    ucl_ssh_hook = ssh_jump_hook.SSHJumpHook(
        ssh_conn_id="ucl_ssh_conn", cmd_timeout=None)
    ucl_sftp_hook = SFTPHook(ssh_hook=ucl_ssh_hook)

    model_tasks_config = common.load_config()["seq_classification"]["models"]

    with TaskGroup(group_id="git") as tg:
        ucl_bbk_mres_git_reset_task = git_tasks.create_git_reset_task(
            "ucl_bbk_mres_git_reset", ucl_ssh_hook, git_branch,
            tasks.UCL_BASE_DIR, hard_reset=True)

        ucl_sge_utils_git_reset_task = git_tasks.create_git_reset_task(
            "ucl_sge_utils_git_reset", ucl_ssh_hook,
            GIT_DEFAULT_SGE_UTILS_BRANCH, tasks.UCL_SGE_UTILS_BASE_DIR,
            hard_reset=True)

    attention_tasks = []
    svm_embeddings_prediction_tasks = []

    (task_check_remove_sim_seqs_train,
     task_remove_sim_seqs_train,
     task_check_remove_sim_seqs_test,
     task_remove_sim_seqs_test) = tasks.create_remove_similar_sequences_tasks(
        chain, git_branch=git_branch)

    with TaskGroup(group_id="adjust_label_counts") as tg:
        (task_check_undersample_test,
         task_undersample_test) = tasks.create_undersample_test_tasks(
            chain, git_branch=git_branch)

    (task_check_split_data,
     task_split_data) = tasks.create_split_data_tasks(
         chain, git_branch=git_branch)

    (task_check_shuffle_labels,
     task_shuffle_labels) = tasks.create_shuffle_labels_tasks(
         chain, git_branch=git_branch)

    task_remove_sim_seqs_train >> task_check_remove_sim_seqs_test
    task_remove_sim_seqs_train >> task_check_shuffle_labels
    task_remove_sim_seqs_train >> task_check_split_data
    # task_undersample_train >> task_check_split_data
    task_split_data >> task_check_undersample_test
    task_remove_sim_seqs_test >> task_check_undersample_test
    # task_remove_sim_seqs_train >> task_check_undersample_train

    ucl_put_input = tasks.create_ucl_upload_sequences_task(
        ucl_sftp_hook, chain)

    task_split_data >> ucl_put_input

    predict_tasks = []

    ucl_external_models_path = Variable.get(VAR_UCL_EXTERNAL_MODELS_PATH, "")

    for model_config in model_tasks_config:
        if not model_config.get("enabled", True):
            continue

        model = model_config["model"]
        model_path = model_config.get("path")
        ucl_cluster = model_config.get("ucl_cluster", False)
        use_default_model_tokenizer = model_config.get(
            "use_default_model_tokenizer")
        use_accelerate = model_config.get("accelerate", False)
        num_gpus = model_config.get(
            "gpus", DEFAULT_GPUS)
        batch_size = model_config.get("batch_size", tasks.DEFAULT_BATCH_SIZE)
        model_path_pt = None
        ucl_model_path = None

        if model_path:
            model_path_pt = os.path.join(EXTERNAL_MODELS_PATH, model_path)
            ucl_model_path = os.path.join(ucl_external_models_path, model_path)

        task_model_name = model_config.get("task_model_name", model)

        with TaskGroup(group_id=task_model_name) as tg:
            if not ucl_cluster:
                with TaskGroup(group_id=f"training") as tg1:
                    (check_update_model,
                     training) = tasks.create_training_tasks(
                        model, chain, model_path_pt,
                        use_default_model_tokenizer, task_model_name,
                        batch_size, use_accelerate, num_gpus, git_branch)

                    task_split_data >> check_update_model
                    last_training_task = training
            else:
                with TaskGroup(group_id=f"ucl_training") as tg1:
                    (check_update_model, ucl_training,
                     get_model_zip, ucl_delete_model_zip,
                     unzip_model) = tasks.create_ucl_training_tasks(
                        ucl_ssh_hook, ucl_sftp_hook, model, chain,
                        ucl_model_path, use_default_model_tokenizer,
                        task_model_name)

                    ucl_bbk_mres_git_reset_task >> ucl_training
                    ucl_sge_utils_git_reset_task >> ucl_training
                    check_update_model >> ucl_put_input
                    ucl_put_input >> ucl_training
                    check_update_model >> ucl_training
                    last_training_task = unzip_model

            with TaskGroup(group_id=f"predict") as tg1:
                (check_update_predict_metrics_pt,
                 predict_metrics_pt) = tasks.create_predict_tasks(
                    model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name,
                    num_gpus=num_gpus, git_branch=git_branch)

                (check_update_predict_metrics_ft,
                 predict_metrics_ft) = tasks.create_predict_tasks(
                    model, chain, None,
                    use_default_model_tokenizer, task_model_name,
                    pre_trained=False, num_gpus=num_gpus,
                    git_branch=git_branch)

            with TaskGroup(group_id=f"attentions") as tg1:
                (check_updated_attentions_pt,
                 attentions_pt) = tasks.create_attention_comparison_tasks(
                    model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name,
                    num_gpus=num_gpus, git_branch=git_branch)

                (check_updated_attentions_ft,
                 attentions_ft) = tasks.create_attention_comparison_tasks(
                    model, chain, None,
                    use_default_model_tokenizer, task_model_name,
                    pre_trained=False, num_gpus=num_gpus,
                    git_branch=git_branch)

            with TaskGroup(group_id=f"embeddings") as tg1:
                (check_updated_embeddings_pt,
                 get_embeddings_pt, check_svm_emb_pred_pt,
                 svm_emb_pred_pt, check_svm_emb_pred_pt_shuffled,
                 svm_emb_pred_pt_shuffled) = tasks.create_embeddings_tasks(
                    model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name,
                    num_gpus=num_gpus, git_branch=git_branch)

                (check_updated_embeddings_ft,
                 get_embeddings_ft, check_svm_emb_pred_ft,
                 svm_emb_pred_ft, check_svm_emb_pred_ft_shuffled,
                 svm_emb_pred_ft_shuffled) = tasks.create_embeddings_tasks(
                    model, chain, None,
                    use_default_model_tokenizer, task_model_name,
                    pre_trained=False, num_gpus=num_gpus,
                    git_branch=git_branch)

            task_shuffle_labels >> [
                check_svm_emb_pred_pt_shuffled,
                check_svm_emb_pred_ft_shuffled]

            task_remove_sim_seqs_train >> [
                check_updated_embeddings_pt,
                check_updated_embeddings_ft]

            task_undersample_test >> [
                check_update_predict_metrics_pt,
                check_update_predict_metrics_ft,
                check_updated_attentions_pt,
                check_updated_attentions_ft]

            last_training_task >> check_update_predict_metrics_ft
            last_training_task >> check_updated_attentions_ft
            last_training_task >> check_updated_embeddings_ft

            predict_tasks.extend([predict_metrics_pt, predict_metrics_ft])
            attention_tasks.extend([attentions_pt, attentions_ft])
            svm_embeddings_prediction_tasks.extend(
                [svm_emb_pred_pt, svm_emb_pred_pt_shuffled,
                 svm_emb_pred_ft, svm_emb_pred_ft_shuffled])

    with TaskGroup(group_id=f"reports") as tg:
        process_attention_comparison_rmd = tasks.create_rmarkdown_task(
            "process_attention_comparison_rmd",
            ATTENTIONS_RMD,
            common.OUTPUT_PATH,
            ATTENTIONS_RMD_OUTPUT_FILENAME,
            chain, git_branch=git_branch)

        process_attention_comparison_rmd << attention_tasks

        process_cv_auroc_rmd = tasks.create_rmarkdown_task(
            "process_cv_auroc_rmd",
            CV_AUROC_RMD,
            common.OUTPUT_PATH,
            CV_AUROC_RMD_OUTPUT_FILENAME,
            chain, git_branch=git_branch)

        process_cv_auroc_rmd << svm_embeddings_prediction_tasks

        process_metrics_rmd = tasks.create_rmarkdown_task(
            "metrics_rmd",
            CV_METRICS_RMD,
            common.OUTPUT_PATH,
            CV_METRICS_RMD_OUTPUT_FILENAME,
            chain, git_branch=git_branch)

        process_metrics_rmd << predict_tasks

        data_url = f"{utils.get_base_url()}/data/"

        send_success_email = EmailOperator(
            task_id="send_success_email",
            to="{{ var.value.email_to }}",
            subject="BBK-MRes sequence classification tasks completed",
            html_content=(
                '<p>'
                'Start time: {{ data_interval_start }}<br/>'
                '</br>'
                'Chain: <b>{{ var.value.chain}}</b></br>'
                'Branch: <b>{{ var.value.bbk_mres_git_branch}}</b></br>'
                '</p>'
                '<h3>Results</h3>'
                '<p>'
                '<ul>'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                'metrics.html">Metrics</a>'
                '  </li>'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                'attention_comparison.html">Attention comparison</a>'
                '  </li>'
                '  <li>'
                '    <a href="{{ params.data_url }}/output/{{ run_id }}/'
                'cv_auroc.html">CV AUROC</a>'
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

        send_success_email << [
            process_attention_comparison_rmd,
            process_cv_auroc_rmd,
            process_metrics_rmd]
