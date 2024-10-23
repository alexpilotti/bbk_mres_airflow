import urllib.parse

BASE_PATH = "/data/bbk-mres"
DATA_PATH = f"{BASE_PATH}/data"
OUTPUT_PATH = f"{DATA_PATH}/output"


def get_dag_run_url(dag_id, run_id):
    return (
        f"http://10.7.231.224:81/dags/{dag_id}/grid?"
        f"dag_run_id={urllib.parse.quote_plus(run_id)}")
