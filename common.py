import json
import os

DATA_PATH = "/data"
EXTERNAL_MODELS_PATH = f"{DATA_PATH}/pre_trained_models"
MODELS_PATH = f"{DATA_PATH}/models"
OUTPUT_PATH = f"{DATA_PATH}/output"

VAR_FROZEN_LAYERS = "frozen_layers"
DEFAULT_FROZEN_LAYERS = -3
VAR_CHAIN = "chain"
VAR_GIT_BBK_MRES_BRANCH = "bbk_mres_git_branch"
GIT_BBK_MRES_DEFAULT_BRANCH = "main"

VAR_ENABLED_MODELS = "enabled_models"

CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"

PRE_TRAINED = "PT"
FINE_TUNED = "FT"

CUDA_CONTAINER_IMAGE = "registry.bbk-mres:5000/bbk-mres-cuda:latest"
R_CONTAINER_IMAGE = "registry.bbk-mres:5000/bbk-mres-r:latest"


def load_config():
    dag_folder = os.path.dirname(__file__)
    config_path = os.path.join(dag_folder, "config.json")

    with open(config_path, "r") as f:
        return json.load(f)
