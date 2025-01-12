import urllib.parse

from airflow.configuration import conf


def get_base_url():
    return conf.get('webserver', 'base_url', fallback='http://localhost:8080')


def get_dag_run_url(dag_id, run_id):
    return (
        f"{get_base_url()}/dags/{dag_id}/grid?"
        f"dag_run_id={urllib.parse.quote_plus(run_id)}")
