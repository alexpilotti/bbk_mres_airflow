import os

import jinja2


from airflow.decorators import task_group
from airflow.models import Variable
from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.providers.ssh.operators.ssh import SSHOperator

from bbk_mres_airflow import common
from bbk_mres_airflow import k8s
from bbk_mres_airflow import fs_compare_operators

UCL_GPU_POOL = "ucl_gpu_pool"

INPUT_PATH = f"{common.DATA_PATH}/S_FULL.parquet"
TEST_INPUT_PATH = (
    f"{common.DATA_PATH}/S_FF_20240729_test.parquet")

REMOVE_SIM_SEQ_OUTPUT_PATH = (
    f"{common.DATA_PATH}/S_filtered" + "_{chain}.parquet")
SHUFFLE_LABELS_OUTPUT_PATH = (
    f"{common.DATA_PATH}/S_shuffled" + "_{chain}{filtered}.parquet")
SPLIT_DATA_OUTPUT_PATH = (
    f"{common.DATA_PATH}/S_split" + "_{chain}{filtered}.parquet")
TRAINING_OUTPUT_PATH = (
    f"{common.MODELS_PATH}/" + "seq_prediction_{model}_{chain}{filtered}/")
TRAINING_OUTPUT_PATH_CHECK = TRAINING_OUTPUT_PATH + "config.json"

REMOVE_SIM_SEQ_TEST_OUTPUT_PATH = (
    f"{common.DATA_PATH}/S_FF_20240729_test_filtered" + "_{chain}.parquet")
UNDERSAMPLE_TEST_OUTPUT_PATH = (
    f"{common.DATA_PATH}/S_FF_20240729_test_filtered_adj" + "_{chain}.parquet")
PREDICT_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "predict_metrics_{model}_{chain}_{pre_trained}{filtered}.json")

ATTENTIONS_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "attention_weights_{model}_{chain}_{pre_trained}{filtered}.parquet")

EMBEDDINGS_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "embeddings_{model}_{chain}_{pre_trained}{filtered}.pt")
EMBEDDINGS_TEST_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "embeddings_{model}_{chain}_{pre_trained}{filtered}_test.pt")

SVM_EMBEDDINGS_MODEL_PATH = (
    f"{common.DATA_PATH}/" +
    "svm_{model}_{chain}_{pre_trained}{filtered}.jbl")
SVM_EMBEDDINGS_BUILD_MODEL_METRICS_PATH = (
    f"{common.DATA_PATH}/" +
    "svm_{model}_{chain}_{pre_trained}{filtered}.csv")
SVM_EMBEDDINGS_PREDICT_METRICS_PATH = (
    f"{common.DATA_PATH}/" +
    "predict_metrics_{model}_{chain}_{pre_trained}_SVM{filtered}.json")
SVM_EMBEDDINGS_SHUFFLED_MODEL_PATH = (
    f"{common.DATA_PATH}/" +
    "svm_{model}_{chain}_{pre_trained}{filtered}_shuffled.jbl")
SVM_EMBEDDINGS_SHUFFLED_BUILD_MODEL_OUTPUT_PATH = (
    f"{common.DATA_PATH}/" +
    "svm_{model}_{chain}_{pre_trained}{filtered}_shuffled.csv")
SVM_EMBEDDINGS_SHUFFLED_PREDICT_METRICS_PATH = (
    f"{common.DATA_PATH}/" +
    "predict_metrics_{model}_{chain}_{pre_trained}" +
    "_SVM_shuffled{filtered}.json")

UCL_SGE_UTILS_BASE_DIR = "/SAN/fraternalilab/bcells/apilotti/sge-utils"

UCL_BASE_DIR = "/SAN/fraternalilab/bcells/apilotti/bbk-mres"
UCL_DATA_PATH = f"{UCL_BASE_DIR}/data"
UCL_MODELS_PATH = f"{UCL_BASE_DIR}/models"
UCL_TRAINING_INPUT_PATH = (
    f"{UCL_DATA_PATH}/S_split" + "_{chain}{filtered}.parquet")
UCL_TRAINING_OUTPUT_PATH = (
    f"{UCL_MODELS_PATH}/" + "seq_prediction_{model}_{chain}{filtered}/")
UCL_DOWNLOAD_TMP_PATH = f"{common.DATA_PATH}/ucl_download_tmp"

UCL_TRAINING_NUM_GPUS = 2
UCL_TRAINING_GPU_TYPE = "a40"

DATASET_TRAIN = "train"

POSITIVE_LABELS = "S+ S1+ S2+"
FOLD = 1

MAX_ATTENTION_SEQUENCES = 200

MIN_SEQ_ID = 0.9

DEFAULT_BATCH_SIZE = 64

DEFAULT_GPUS = 2

COMMON_CMD = (
    "git fetch && git reset --hard origin/{{ params.git_branch }} && "
    "{% if params.accelerate %}"
    "accelerate launch --config_file /data/accelerate.yaml "
    "--num_processes=$(nvidia-smi --list-gpus | wc -l)"
    "{% else %}python{% endif %} "
    "attention_comparison/cli.py ")

REMOVE_SIMILAR_SEQUENCES_CMD = COMMON_CMD + (
    "remove-similar-sequences -i {{ params.input }} -o {{ params.output }} "
    "{% if params.target %} -t {{ params.target }} {% endif %}"
    "-c {{ params.chain }} -m {{ params.min_seq_id }}")

SPLIT_DATA_CMD = COMMON_CMD + (
    "split-data -i {{ params.input }} -o {{ params.output }} "
    "-l {{ params.positive_labels }} -f {{ params.fold }}")

UNDERSAMPLE_CMD = COMMON_CMD + (
    "undersample -i {{ params.input }} -o {{ params.output }}"
    "{% if params.target %} -t {{ params.target }}{% endif %}"
    "{% if params.target_dataset %} --target-dataset "
    "{{ params.target_dataset }}{% endif %}")

TRAINING_CMD = COMMON_CMD + (
    "seq-fine-tuning -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }} -b {{ params.batch_size}}"
    "{% if params.frozen_layers %} --frozen-layers "
    "{{ params.frozen_layers }}{% endif %}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

PREDICT_CMD = COMMON_CMD + (
    "seq-prediction -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

ATTENTIONS_CMD = COMMON_CMD + (
    "attentions -m {{ params.model }} -i {{ params.input }} "
    "-o {{ params.output }} -c {{ params.chain }}"
    "{% if params.max_sequences %} --max-sequences {{ params.max_sequences }}"
    "{% endif %}{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

EMBEDDINGS_CMD = COMMON_CMD + (
    "embeddings -m {{ params.model }} "
    "-i {{ params.input }} -o {{ params.output }} "
    "-c {{ params.chain }}"
    "{% if params.model_path %} -p {{ params.model_path }}"
    "{% endif %}{% if params.use_default_model_tokenizer %} "
    "--use-default-model-tokenizer"
    "{% endif %}")

SHUFFLE_CMD = COMMON_CMD + (
    "shuffle -i {{ params.input }} -o {{ params.output }}")

SVM_EMBEDDINGS_BUILD_MODEL_CMD = COMMON_CMD + (
    "svm-embeddings-build-model -i {{ params.input }} "
    "-e {{ params.embeddings }} -m {{ params.model }} "
    "-o {{ params.output_metrics }} -l {{ params.positive_labels }}")

SVM_EMBEDDINGS_PREDICT_CMD = COMMON_CMD + (
    "svm-embeddings-predict -i {{ params.input }} "
    "-e {{ params.embeddings }} -m {{ params.model }} "
    "-o {{ params.output_metrics }}")

RMARKDOWN_CMD = (
    "git fetch && git reset --hard origin/{{ params.git_branch }} && "
    "mkdir -p '{{ params.output_base_dir }}/{{ run_id }}' && "
    "/usr/bin/Rscript -e "
    "\"rmarkdown::render('{{ params.rmd_path }}', "
    "output_file = '{{ params.output_base_dir }}/{{ run_id }}/"
    "{{ params.output_filename }}', "
    "params = list({{ params.params }}))\"")

SGE_CMD = (
    f"{UCL_SGE_UTILS_BASE_DIR}/"
    "run_sge_task.sh {{ params.job_name }} \"{{ params.native_specs }}\" "
    "{{ params.cmd }}")

SGE_NATIVE_SPECS = (
    "-l h_rt=19:00:00 -R y -l tmem={{ mem_gb }}G "
    "{% if num_gpus > 0 %}-l gpu=True -pe gpu {{ num_gpus }} {% endif %}"
    "{% if gpu_type %}-l gpu_type={{ gpu_type }} {% endif %}"
    "-l tscratch={{ scratch_gb }}G")

UCL_TRAINING_CMD = (
    f"{UCL_BASE_DIR}/" + "fine_tuning.sh {{ model }} "
    "{{ chain }} {{ input }} {{ output }} {{ model_path or '' }} "
    "{{ use_default_model_tokenizer or '' }}")


def create_attention_comparison_tasks(
        model, chain, model_path=None, use_default_model_tokenizer=None,
        task_model_name=None, pre_trained=True, filtered=True,
        num_gpus=DEFAULT_GPUS, use_accelerate=False, git_branch="main"):
    pre_trained_str = (
        common.PRE_TRAINED if pre_trained else common.FINE_TUNED)
    filtered_str = _get_filtered_str(filtered)

    if filtered:
        input_path = UNDERSAMPLE_TEST_OUTPUT_PATH.format(chain=chain)
    else:
        input_path = TEST_INPUT_PATH

    output_path = ATTENTIONS_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str,
        filtered=filtered_str)
    training_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain, filtered=filtered_str)
    max_sequences = Variable.get("max_attention_sequences",
                                 default_var=MAX_ATTENTION_SEQUENCES)

    check_inputs = [input_path]
    if not pre_trained:
        model_path = TRAINING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain, filtered=filtered_str)
        check_inputs.append(training_path_check)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=("check_updated_attentions_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        path1=check_inputs,
        path2=output_path,
        trigger_rule="none_failed"
    )

    task_attention_comparison = k8s.create_pod_operator(
        task_id=("attention_comparison_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=ATTENTIONS_CMD,
        params={"model": model,
                "input": input_path,
                "output": output_path,
                "max_sequences": max_sequences,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check >> task_attention_comparison

    return task_check, task_attention_comparison


@task_group(group_id="remove_similar_sequences")
def create_remove_similar_sequences_tasks(chain, use_accelerate=False,
                                          git_branch="main"):
    train_input_path = INPUT_PATH
    train_output_path = REMOVE_SIM_SEQ_OUTPUT_PATH.format(chain=chain)
    test_target_path = train_output_path
    test_input_path = TEST_INPUT_PATH
    test_output_path = REMOVE_SIM_SEQ_TEST_OUTPUT_PATH.format(chain=chain)

    task_check_train = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_remove_similar_sequences_train",
        path1=train_input_path,
        path2=train_output_path,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_remove_similar_sequences_train = k8s.create_pod_operator(
        task_id=f"remove_similar_sequences_train",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=REMOVE_SIMILAR_SEQUENCES_CMD,
        params={"input": train_input_path,
                "output": train_output_path,
                "chain": chain,
                "target": None,
                "min_seq_id": MIN_SEQ_ID,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check_train >> task_remove_similar_sequences_train

    task_check_test = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_remove_similar_sequences_test",
        path1=[test_input_path, test_target_path],
        path2=test_output_path,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_remove_similar_sequences_test = k8s.create_pod_operator(
        task_id=f"remove_similar_sequences_test",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=REMOVE_SIMILAR_SEQUENCES_CMD,
        params={"input": test_input_path,
                "target": test_target_path,
                "output": test_output_path,
                "chain": chain,
                "min_seq_id": MIN_SEQ_ID,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_remove_similar_sequences_train >> task_check_test
    task_check_test >> task_remove_similar_sequences_test

    return (task_check_train, task_remove_similar_sequences_train,
            task_check_test, task_remove_similar_sequences_test)


def create_undersample_test_tasks(chain, use_accelerate=False,
                                  git_branch="main"):
    training_input_path = SPLIT_DATA_OUTPUT_PATH.format(
        chain=chain, filtered=_get_filtered_str(False))
    undersample_test_input = REMOVE_SIM_SEQ_TEST_OUTPUT_PATH.format(
        chain=chain)
    undersample_test_output = UNDERSAMPLE_TEST_OUTPUT_PATH.format(chain=chain)

    task_check_test = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_undersample_test",
        path1=[undersample_test_input, training_input_path],
        path2=undersample_test_output,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_undersample_test = k8s.create_pod_operator(
        task_id=f"undersample_test",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=UNDERSAMPLE_CMD,
        params={"input": undersample_test_input,
                "output": undersample_test_output,
                "target": training_input_path,
                "target_dataset": DATASET_TRAIN,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check_test >> task_undersample_test

    return (task_check_test, task_undersample_test)


def _get_filtered_str(filtered):
    return "" if filtered else "_NF"


def create_split_data_tasks(chain, filtered, use_accelerate=False,
                            git_branch="main"):
    filtered_str = _get_filtered_str(filtered)
    if filtered:
        input_path = REMOVE_SIM_SEQ_OUTPUT_PATH.format(chain=chain)
    else:
        input_path = INPUT_PATH.format(chain=chain)
    output_path = SPLIT_DATA_OUTPUT_PATH.format(
        chain=chain, filtered=filtered_str)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_split_data{filtered_str}",
        path1=input_path,
        path2=output_path,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_split_data = k8s.create_pod_operator(
        task_id=f"split_data{filtered_str}",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=SPLIT_DATA_CMD,
        params={"input": input_path,
                "output": output_path,
                "positive_labels": POSITIVE_LABELS,
                "fold": FOLD,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check >> task_split_data

    return task_check, task_split_data


def create_shuffle_labels_tasks(chain, use_accelerate=False, filtered=True,
                                git_branch="main"):
    filtered_str = _get_filtered_str(filtered)

    if filtered:
        input_path = REMOVE_SIM_SEQ_OUTPUT_PATH.format(chain=chain)
    else:
        input_path = INPUT_PATH

    output_path = SHUFFLE_LABELS_OUTPUT_PATH.format(
        chain=chain, filtered=filtered_str)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_shuffle_data{filtered_str}",
        path1=input_path,
        path2=output_path,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_shuffle_labels = k8s.create_pod_operator(
        task_id=f"shuffle_labels{filtered_str}",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=SHUFFLE_CMD,
        params={"input": input_path,
                "output": output_path,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check >> task_shuffle_labels

    return task_check, task_shuffle_labels


def create_training_tasks(model, chain, model_path=None,
                          use_default_model_tokenizer=None,
                          task_model_name=None, batch_size=DEFAULT_BATCH_SIZE,
                          filtered=True, use_accelerate=False,
                          num_gpus=DEFAULT_GPUS, git_branch="main",
                          frozen_layers=None):
    filtered_str = _get_filtered_str(filtered)

    input_path = SPLIT_DATA_OUTPUT_PATH.format(
        chain=chain, filtered=filtered_str)
    output_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain, filtered=filtered_str)
    output_path = TRAINING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, filtered=filtered_str)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_update_model_{task_model_name}_{chain}",
        path1=input_path,
        path2=output_path_check,
        trigger_rule="none_failed"
    )

    task_train = k8s.create_pod_operator(
        task_id=f"training_{task_model_name}_{chain}",
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=TRAINING_CMD,
        params={"model": model,
                "input": input_path,
                "output": output_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "batch_size": batch_size,
                "accelerate": use_accelerate,
                "git_branch": git_branch,
                "frozen_layers": frozen_layers},
    )

    task_check >> task_train

    return task_check, task_train


def create_predict_tasks(model, chain, model_path=None,
                         use_default_model_tokenizer=None,
                         task_model_name=None, pre_trained=True,
                         filtered=True, num_gpus=DEFAULT_GPUS,
                         use_accelerate=False, git_branch="main"):
    pre_trained_str = (
        common.PRE_TRAINED if pre_trained else common.FINE_TUNED)
    filtered_str = _get_filtered_str(filtered)

    if filtered:
        input_path = UNDERSAMPLE_TEST_OUTPUT_PATH.format(chain=chain)
    else:
        input_path = TEST_INPUT_PATH

    output_path = PREDICT_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str,
        filtered=filtered_str)

    check_inputs = [input_path]

    if not pre_trained:
        model_path = TRAINING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain, filtered=filtered_str)
        training_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
            model=task_model_name, chain=chain, filtered=filtered_str)
        check_inputs.append(training_path_check)

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=(f"check_update_predict_metrics_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        path1=check_inputs,
        path2=output_path,
        trigger_rule="none_failed"
    )

    task_predict = k8s.create_pod_operator(
        task_id=(f"predict_{task_model_name}_{chain}_{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=PREDICT_CMD,
        params={"model": model,
                "input": input_path,
                "output": output_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check >> task_predict

    return task_check, task_predict


def create_embeddings_tasks(model, chain, model_path=None,
                            use_default_model_tokenizer=None,
                            task_model_name=None, pre_trained=True,
                            filtered=True, num_gpus=DEFAULT_GPUS,
                            use_accelerate=False, git_branch="main"):
    pre_trained_str = (
        common.PRE_TRAINED if pre_trained else common.FINE_TUNED)
    filtered_str = _get_filtered_str(filtered)

    embeddings_path = EMBEDDINGS_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str,
        filtered=filtered_str)

    if filtered:
        input_path = REMOVE_SIM_SEQ_OUTPUT_PATH.format(chain=chain)
        test_input_path = UNDERSAMPLE_TEST_OUTPUT_PATH.format(chain=chain)
    else:
        input_path = INPUT_PATH
        test_input_path = TEST_INPUT_PATH

    test_embeddings_path = EMBEDDINGS_TEST_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str,
        filtered=filtered_str)

    check_inputs = [input_path]
    check_test_inputs = [test_input_path]

    if not pre_trained:
        model_path = TRAINING_OUTPUT_PATH.format(
            model=task_model_name, chain=chain, filtered=filtered_str)

        training_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
            model=task_model_name, chain=chain, filtered=filtered_str)
        check_inputs.append(training_path_check)
        check_test_inputs.append(training_path_check)

    task_check_emb = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=("check_update_embeddings_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        path1=check_inputs,
        path2=embeddings_path,
        trigger_rule="none_failed"
    )

    task_embeddings = k8s.create_pod_operator(
        task_id=(
            f"get_embeddings_{task_model_name}_{chain}_{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=EMBEDDINGS_CMD,
        params={"model": model,
                "input": input_path,
                "output": embeddings_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check_emb >> task_embeddings

    task_check_emb_predict = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=("check_update_embeddings_predict_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        path1=check_test_inputs,
        path2=test_embeddings_path,
        trigger_rule="none_failed"
    )

    task_embeddings_predict = k8s.create_pod_operator(
        task_id=(f"get_embeddings_predict_{task_model_name}_{chain}_"
                 f"{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=num_gpus,
        command=EMBEDDINGS_CMD,
        params={"model": model,
                "input": test_input_path,
                "output": test_embeddings_path,
                "chain": chain,
                "model_path": model_path,
                "use_default_model_tokenizer": use_default_model_tokenizer,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_check_emb_predict >> task_embeddings_predict

    svm_model_path = SVM_EMBEDDINGS_MODEL_PATH.format(
        model=task_model_name, chain=chain, pre_trained=pre_trained_str,
        filtered=filtered_str)

    svm_build_model_metrics_output_path = (
        SVM_EMBEDDINGS_BUILD_MODEL_METRICS_PATH.format(
            model=task_model_name, chain=chain, pre_trained=pre_trained_str,
            filtered=filtered_str))

    task_check_svm_build = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=("check_update_svm_embeddings_build_model_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        path1=[input_path, embeddings_path],
        path2=[svm_model_path, svm_build_model_metrics_output_path],
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_svm_embeddings_build_model = k8s.create_pod_operator(
        task_id=("svm_embeddings_build_model_" +
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=SVM_EMBEDDINGS_BUILD_MODEL_CMD,
        params={"input": input_path,
                "model": svm_model_path,
                "output_metrics": svm_build_model_metrics_output_path,
                "embeddings": embeddings_path,
                "positive_labels": POSITIVE_LABELS,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    svm_predict_metrics_output_path = (
        SVM_EMBEDDINGS_PREDICT_METRICS_PATH.format(
            model=task_model_name, chain=chain, pre_trained=pre_trained_str,
            filtered=filtered_str))

    task_check_svm_predict = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=("check_update_svm_embeddings_predict_"
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        path1=[test_input_path, test_embeddings_path, svm_model_path],
        path2=svm_predict_metrics_output_path,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_svm_embeddings_predict = k8s.create_pod_operator(
        task_id=("svm_embeddings_predict_" +
                 f"{task_model_name}_{chain}_{pre_trained_str}"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=SVM_EMBEDDINGS_PREDICT_CMD,
        params={"input": test_input_path,
                "model": svm_model_path,
                "output_metrics": svm_predict_metrics_output_path,
                "embeddings": test_embeddings_path,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    svm_input_path_shuffled = (
        SHUFFLE_LABELS_OUTPUT_PATH.format(chain=chain, filtered=filtered_str))
    svm_model_path_shuffled = (
        SVM_EMBEDDINGS_SHUFFLED_MODEL_PATH.format(
            model=task_model_name, chain=chain, pre_trained=pre_trained_str,
            filtered=filtered_str))
    svm_output_path_shuffled = (
        SVM_EMBEDDINGS_SHUFFLED_BUILD_MODEL_OUTPUT_PATH.format(
            model=task_model_name, chain=chain, pre_trained=pre_trained_str,
            filtered=filtered_str))
    svm_predict_metrics_output_path_shuffled = (
        SVM_EMBEDDINGS_SHUFFLED_PREDICT_METRICS_PATH.format(
            model=task_model_name, chain=chain, pre_trained=pre_trained_str,
            filtered=filtered_str))

    task_check_svm_shuffled = (
        fs_compare_operators.ComparePathDatetimesSensor(
            task_id=("check_update_svm_embeddings_prediction_"
                     f"{task_model_name}_{chain}_{pre_trained_str}_shuffled"),
            path1=[svm_input_path_shuffled, embeddings_path],
            path2=[svm_model_path_shuffled, svm_output_path_shuffled],
            trigger_rule="none_failed"
        ))

    # This task doesn't use CUDA
    task_svm_embeddings_build_model_shuffled = k8s.create_pod_operator(
        task_id=("svm_embeddings_build_model_" +
                 f"{task_model_name}_{chain}_{pre_trained_str}_shuffled"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=SVM_EMBEDDINGS_BUILD_MODEL_CMD,
        params={"input": svm_input_path_shuffled,
                "model": svm_model_path_shuffled,
                "output_metrics": svm_output_path_shuffled,
                "embeddings": embeddings_path,
                "positive_labels": POSITIVE_LABELS,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_chk_svm_pred_shuff = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=("check_update_svm_embeddings_predict_"
                 f"{task_model_name}_{chain}_{pre_trained_str}_shuffled"),
        path1=[test_input_path, test_embeddings_path, svm_model_path_shuffled],
        path2=svm_predict_metrics_output_path_shuffled,
        trigger_rule="none_failed"
    )

    # This task doesn't use CUDA
    task_svm_embeddings_predict_shuffled = k8s.create_pod_operator(
        task_id=("svm_embeddings_predict_" +
                 f"{task_model_name}_{chain}_{pre_trained_str}_shuffled"),
        image=common.CUDA_CONTAINER_IMAGE,
        num_gpus=0,
        command=SVM_EMBEDDINGS_PREDICT_CMD,
        params={"input": test_input_path,
                "model": svm_model_path_shuffled,
                "output_metrics": svm_predict_metrics_output_path_shuffled,
                "embeddings": test_embeddings_path,
                "accelerate": use_accelerate,
                "git_branch": git_branch},
    )

    task_svm_embeddings_build_model >> task_check_svm_predict
    task_check_svm_predict >> task_svm_embeddings_predict
    task_check_svm_build >> task_svm_embeddings_build_model

    task_svm_embeddings_build_model_shuffled >> task_chk_svm_pred_shuff
    task_chk_svm_pred_shuff >> task_svm_embeddings_predict_shuffled
    task_check_svm_shuffled >> task_svm_embeddings_build_model_shuffled

    task_embeddings >> [task_check_svm_build, task_check_svm_shuffled]
    task_embeddings_predict >> [
        task_check_svm_predict, task_chk_svm_pred_shuff]

    return (task_check_emb, task_embeddings, task_check_emb_predict,
            task_embeddings_predict, task_check_svm_build,
            task_svm_embeddings_build_model, task_check_svm_predict,
            task_svm_embeddings_predict, task_check_svm_shuffled,
            task_svm_embeddings_build_model_shuffled,
            task_chk_svm_pred_shuff,
            task_svm_embeddings_predict_shuffled)


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


def _create_grid_engine_task(
        ssh_hook, task_id, cmd, mem_gb=4, num_gpus=DEFAULT_GPUS, gpu_type=None,
        scratch_gb=50, trigger_rule="all_success", retries=5,
        pool="default_pool"):

    template = jinja2.Environment().from_string(SGE_NATIVE_SPECS)
    native_specs = template.render(
        mem_gb=mem_gb, num_gpus=num_gpus, gpu_type=gpu_type,
        scratch_gb=scratch_gb)

    return SSHOperator(
        task_id=task_id,
        ssh_hook=ssh_hook,
        command=SGE_CMD,
        trigger_rule=trigger_rule,
        retries=retries,
        params={"job_name": task_id,
                "native_specs": native_specs,
                "cmd": cmd},
        pool=pool
    )


def create_ucl_upload_sequences_task(ucl_sftp_hook, chain, filtered):
    filtered_str = _get_filtered_str(filtered)
    return SFTPOperator(
        task_id=f"ucl_upload_input_sequences{filtered_str}",
        sftp_hook=ucl_sftp_hook,
        local_filepath=SPLIT_DATA_OUTPUT_PATH.format(
            chain=chain, filtered=filtered_str),
        remote_filepath=UCL_TRAINING_INPUT_PATH.format(
            chain=chain, filtered=filtered_str),
        operation="put",
        create_intermediate_dirs=False)


def create_ucl_training_tasks(
        ucl_ssh_hook, ucl_sftp_hook, model, chain,
        model_path=None, use_default_model_tokenizer=None,
        task_model_name=None, filtered=True):
    filtered_str = _get_filtered_str(filtered)

    input_path = SPLIT_DATA_OUTPUT_PATH.format(
        chain=chain, filtered=filtered_str)
    output_path_check = TRAINING_OUTPUT_PATH_CHECK.format(
        model=task_model_name, chain=chain, filtered=filtered_str)
    output_path = TRAINING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, filtered=filtered_str)

    ucl_input_path = UCL_TRAINING_INPUT_PATH.format(
        chain=chain, filtered=filtered_str)
    ucl_output_path = UCL_TRAINING_OUTPUT_PATH.format(
        model=task_model_name, chain=chain, filtered=filtered_str)

    ucl_zip_path = f"{os.path.normpath(ucl_output_path)}.zip"
    zip_path = os.path.join(
        UCL_DOWNLOAD_TMP_PATH,
        f"{os.path.basename(os.path.normpath(output_path))}.zip")

    task_check = fs_compare_operators.ComparePathDatetimesSensor(
        task_id=f"check_update_model_{task_model_name}_{chain}",
        path1=input_path,
        path2=output_path_check,
        trigger_rule="none_failed"
    )

    template = jinja2.Environment().from_string(UCL_TRAINING_CMD)
    cmd = template.render(
        model=model, chain=chain, input=ucl_input_path,
        output=ucl_output_path, model_path=model_path,
        use_default_model_tokenizer=use_default_model_tokenizer)

    training_task_id = f"ucl_training_{task_model_name}_{chain}"
    training_task = _create_grid_engine_task(
        ucl_ssh_hook, training_task_id, cmd,
        num_gpus=UCL_TRAINING_NUM_GPUS,
        gpu_type=UCL_TRAINING_GPU_TYPE,
        # trigger_rule="none_failed",
        pool=UCL_GPU_POOL)

    task_get_model_zip = SFTPOperator(
        task_id=f"ucl_download_zip_{task_model_name}_{chain}",
        sftp_hook=ucl_sftp_hook,
        local_filepath=zip_path,
        remote_filepath=ucl_zip_path,
        operation="get",
        create_intermediate_dirs=False)

    task_ucl_delete_model_zip = SSHOperator(
        task_id=f"ucl_delete_zip_{task_model_name}_{chain}",
        ssh_hook=ucl_ssh_hook,
        command="rm {{ params.zip_path }}",
        params={"zip_path": ucl_zip_path})

    task_unzip_model = k8s.create_pod_operator(
        task_id=f"unzip_{task_model_name}_{chain}",
        image=common.CUDA_CONTAINER_IMAGE,
        command=("unzip -o {{ params.zip_path }} -d {{ params.models_path }} "
                 "&& rm {{ params.zip_path }}"),
        params={"zip_path": zip_path,
                "models_path": common.MODELS_PATH})

    task_check >> training_task >> task_get_model_zip
    task_get_model_zip >> task_ucl_delete_model_zip >> task_unzip_model

    return (
        task_check, training_task, task_get_model_zip,
        task_ucl_delete_model_zip, task_unzip_model)
