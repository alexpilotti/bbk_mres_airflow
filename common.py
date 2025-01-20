import json
import os

DATA_PATH = "/data"
OUTPUT_PATH = f"{DATA_PATH}/output"

CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"

CUDA_CONTAINER_IMAGE = "registry.bbk-mres:5000/bbk-mres-cuda:latest"
R_CONTAINER_IMAGE = "registry.bbk-mres:5000/bbk-mres-r:latest"


def load_config():
    dag_folder = os.path.dirname(__file__)
    config_path = os.path.join(dag_folder, "config.json")

    with open(config_path, "r") as f:
        return json.load(f)
