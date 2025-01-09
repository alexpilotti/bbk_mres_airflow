import urllib.parse


def get_dag_run_url(dag_id, run_id):
    return (
        f"https://bbk-mres.pilotti.it/dags/{dag_id}/grid?"
        f"dag_run_id={urllib.parse.quote_plus(run_id)}")
