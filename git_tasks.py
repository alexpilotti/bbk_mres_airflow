from airflow.providers.ssh.operators.ssh import SSHOperator


GIT_RESET_CMD = (
    'cd "{{ params.local_path }}" && '
    'git fetch && git reset origin/{{ params.branch }} '
    '{% if params.hard_reset %}--hard{% else %}--soft{% endif %}')


def create_git_reset_task(task_id, ssh_hook, branch, local_path,
                          hard_reset=True):
    return SSHOperator(
        task_id=task_id,
        ssh_hook=ssh_hook,
        command=GIT_RESET_CMD,
        params={"branch": branch,
                "local_path": local_path,
                "hard_reset": hard_reset})
