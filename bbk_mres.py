from datetime import datetime, timedelta

from airflow.models.dag import DAG

from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.providers.sftp.hooks.sftp import SFTPHook
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.utils.task_group import TaskGroup

from bbk_mres_airflow import common
from bbk_mres_airflow import git_tasks
from bbk_mres_airflow import ssh_jump_hook
from bbk_mres_airflow import tasks

VAR_GIT_BBK_MRES_BRANCH = "bbk_mres_git_branch"
GIT_BBK_MRES_DEFAULT_BRANCH = "main"
GIT_DEFAULT_SGE_UTILS_BRANCH = "master"

MODEL_BALM_PAIRED = "BALM-paired"
MODEL_ANTIBERTY = "AntiBERTy"
MODEL_ANTIBERTA2 = "AntiBERTa2"
MODEL_ESM2_650M = "ESM2-650M"
MODEL_ESM2_150M = "ESM2-150M"
MODEL_ESM2_35M = "ESM2-35M"
MODEL_ESM2_8M = "ESM2-8M"

MODEL_NAME_FT_ESM2 = "ft-ESM2"

CHAIN_H = "H"

ATTENTIONS_RMD = f"{common.BASE_PATH}/attention_comparison.Rmd"
ATTENTIONS_RMD_OUTPUT_FILENAME = "attention_comparison.html"

CV_AUROC_RMD = f"{common.BASE_PATH}/cv_auroc.Rmd"
CV_AUROC_RMD_OUTPUT_FILENAME = "cv_auroc.html"

EXTERNAL_MODELS_PATH = "~/kleinstein-lab-projects/Wang2024/models"
BALM_MODEL_PATH = f"{
    EXTERNAL_MODELS_PATH}/BALM-paired_LC-coherence_90-5-5-split_122222/"
FT_ESM2_MODEL_PATH = f"{EXTERNAL_MODELS_PATH}/ESM2-650M_paired-fine-tuning/"

UCL_EXTERNAL_MODELS_PATH = (
    "/SAN/fraternalilab/bcells/apilotti/kleinstein-lab-projects/"
    "Wang2024/models")
UCL_FT_ESM2_MODEL_PATH = (
    f"{UCL_EXTERNAL_MODELS_PATH}/ESM2-650M_paired-fine-tuning/")

DATA_URL = "https://bbk-mres.pilotti.it/data"


with DAG(
    "BBK-MRes-attention-comparison",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
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
    description="BBK-MRes DAG",
    schedule_interval=None,
    # schedule=timedelta(days=1),
    start_date=datetime(2024, 8, 1),
    catchup=False,
    tags=["bbk"],
) as dag:

    ssh_hook = SSHHook(ssh_conn_id="ssh_conn", cmd_timeout=None)
    sftp_hook = SFTPHook("sftp_conn")

    ucl_ssh_hook = ssh_jump_hook.SSHJumpHook(
        ssh_conn_id="ucl_ssh_conn", cmd_timeout=None)
    ucl_sftp_hook = SFTPHook(ssh_hook=ucl_ssh_hook)

    task_info = [
        (False, MODEL_ANTIBERTY, CHAIN_H, None, None, False, None),
        (False, MODEL_ANTIBERTA2, CHAIN_H, None, None, False, None),
        (False, MODEL_BALM_PAIRED, CHAIN_H, BALM_MODEL_PATH, None, False,
         None),
        (False, MODEL_ESM2_8M, CHAIN_H, None, None, False, None),
        (False, MODEL_ESM2_35M, CHAIN_H, None, None, False, None),
        (False, MODEL_ESM2_150M, CHAIN_H, None, None, False, None),
        (True, MODEL_ESM2_650M, CHAIN_H, None, None, False, None),
        (True, MODEL_ESM2_650M, CHAIN_H, FT_ESM2_MODEL_PATH,
         UCL_FT_ESM2_MODEL_PATH, True, MODEL_NAME_FT_ESM2)
    ]

    with TaskGroup(group_id="git") as tg:
        git_branch = Variable.get(
            VAR_GIT_BBK_MRES_BRANCH, default_var=GIT_BBK_MRES_DEFAULT_BRANCH)
        ucl_bbk_mres_git_reset_task = git_tasks.create_git_reset_task(
            "ucl_bbk_mres_git_reset", ucl_ssh_hook, git_branch,
            tasks.UCL_BASE_DIR, hard_reset=True)

        ucl_sge_utils_git_reset_task = git_tasks.create_git_reset_task(
            "ucl_sge_utils_git_reset", ucl_ssh_hook,
            GIT_DEFAULT_SGE_UTILS_BRANCH, tasks.UCL_SGE_UTILS_BASE_DIR,
            hard_reset=True)

    attention_tasks = []
    svm_embeddings_prediction_tasks = []

    (task_check_remove_similar_seqs,
     task_remove_similar_seqs) = tasks.create_remove_similar_sequences_tasks(
        ssh_hook, sftp_hook, CHAIN_H)

    (task_check_split_data,
     task_split_data) = tasks.create_split_data_tasks(
         ssh_hook, sftp_hook)

    task_remove_similar_seqs >> task_check_split_data

    (get_tmp_input,
     ucl_put_input) = tasks.create_ucl_upload_sequences_task(
         sftp_hook, ucl_sftp_hook)

    task_split_data >> get_tmp_input

    for (ucl_cluster, model, chain, model_path_pt, ucl_model_path,
         use_default_model_tokenizer, task_model_name) in task_info:
        if not task_model_name:
            task_model_name = model

        with TaskGroup(group_id=task_model_name) as tg:
            if not ucl_cluster:
                with TaskGroup(group_id=f"training") as tg1:
                    (check_update_model,
                     training) = tasks.create_training_tasks(
                        ssh_hook, sftp_hook, model, chain, model_path_pt,
                        use_default_model_tokenizer, task_model_name)

                    task_split_data >> check_update_model
                    last_training_task = training
            else:
                with TaskGroup(group_id=f"ucl_training") as tg1:
                    (check_update_model, ucl_training,
                     get_model_zip, ucl_delete_model_zip, put_model_zip,
                     unzip_model) = tasks.create_ucl_training_tasks(
                        ucl_ssh_hook, ucl_sftp_hook, ssh_hook, sftp_hook,
                        model, chain, ucl_model_path,
                        use_default_model_tokenizer, task_model_name)

                    ucl_bbk_mres_git_reset_task >> ucl_training
                    ucl_sge_utils_git_reset_task >> ucl_training
                    check_update_model >> ucl_put_input
                    ucl_put_input >> ucl_training
                    check_update_model >> ucl_training
                    last_training_task = unzip_model

            with TaskGroup(group_id=f"attentions") as tg1:
                (check_updated_attentions_pt,
                 attentions_pt) = tasks.create_attention_comparison_tasks(
                    ssh_hook, sftp_hook, model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name)

                (check_updated_attentions_ft,
                 attentions_ft) = tasks.create_attention_comparison_tasks(
                    ssh_hook, sftp_hook, model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name,
                    pre_trained=False)

            with TaskGroup(group_id=f"embeddings") as tg1:
                (check_updated_embeddings_pt,
                 get_embeddings_pt, check_svm_emb_pred_pt,
                 svm_emb_pred_pt, check_svm_emb_pred_pt_shuffled,
                 svm_emb_pred_pt_shuffled) = tasks.create_embeddings_tasks(
                    ssh_hook, sftp_hook, model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name)

                (check_updated_embeddings_ft,
                 get_embeddings_ft, check_svm_emb_pred_ft,
                 svm_emb_pred_ft, check_svm_emb_pred_ft_shuffled,
                 svm_emb_pred_ft_shuffled) = tasks.create_embeddings_tasks(
                    ssh_hook, sftp_hook, model, chain, model_path_pt,
                    use_default_model_tokenizer, task_model_name,
                    pre_trained=False)

                last_training_task >> check_updated_attentions_ft
                last_training_task >> check_updated_embeddings_ft

            attention_tasks.extend([attentions_pt, attentions_ft])

            svm_embeddings_prediction_tasks.extend(
                [svm_emb_pred_pt, svm_emb_pred_pt_shuffled,
                 svm_emb_pred_ft, svm_emb_pred_ft_shuffled])

    with TaskGroup(group_id=f"reports") as tg:
        process_attention_comparison_rmd = tasks.create_rmarkdown_task(
            ssh_hook,
            "process_attention_comparison_rmd",
            ATTENTIONS_RMD,
            common.OUTPUT_PATH,
            ATTENTIONS_RMD_OUTPUT_FILENAME,
            CHAIN_H)

        process_attention_comparison_rmd << attention_tasks

        process_cv_auroc_rmd = tasks.create_rmarkdown_task(
            ssh_hook,
            "process_cv_auroc_rmd",
            CV_AUROC_RMD,
            common.OUTPUT_PATH,
            CV_AUROC_RMD_OUTPUT_FILENAME,
            CHAIN_H)

        process_cv_auroc_rmd << svm_embeddings_prediction_tasks

        send_success_email = EmailOperator(
            task_id="send_success_email",
            to="{{ var.value.email_to }}",
            subject="BBK-MRes tasks completed",
            html_content=(
                '<p>'
                'Start time: {{ data_interval_start }}<br/>'
                '</p>'
                '<h3>Results</h3>'
                '<ul>'
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
                "data_url": DATA_URL,
                "get_dag_run_url": common.get_dag_run_url}
            )

        send_success_email << [
            process_attention_comparison_rmd, process_cv_auroc_rmd]
