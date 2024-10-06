import logging

from airflow.providers.ssh.hooks import ssh as ssh_hooks
from airflow.providers.ssh.operators import ssh as ssh_operators

from bbk_mres_airflow import openstack

logger = logging.getLogger(__name__)


class OpenStackSSHOperator(ssh_operators.SSHOperator):
    def __init__(self,
                 auth_url: str,
                 username: str,
                 password: str,
                 project_name: str,
                 user_domain_name: str,
                 project_domain_name: str,
                 instance_name: str,
                 image_name: str,
                 flavor_name: str,
                 keypair_name: str,
                 network_name: str,
                 volume_size_gb: int,
                 floating_network_name: str,
                 server_username: str,
                 key_file: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._auth_url = auth_url
        self._username = username
        self._password = password
        self._project_name = project_name
        self._user_domain_name = user_domain_name
        self._project_domain_name = project_domain_name
        self._instance_name = instance_name
        self._image_name = image_name
        self._flavor_name = flavor_name
        self._keypair_name = keypair_name
        self._network_name = network_name
        self._volume_size_gb = volume_size_gb
        self._floating_network_name = floating_network_name
        self._server_username = server_username
        self._key_file = key_file

    def execute(self, context):
        conn = openstack.get_connection(
            self._auth_url, self._username, self._password, self._project_name,
            self._user_domain_name, self._project_domain_name)

        fip = openstack.get_floating_ip(conn, self._floating_network_name)

        server = openstack.create_server(
            conn, self._instance_name, self._image_name, self._flavor_name,
            self._keypair_name, self._network_name, self._volume_size_gb)

        try:
            openstack.attach_floating_ip(conn, server.id, fip)
            remote_address = fip.floating_ip_address

            logger.info(f"Server \"{server.name}\" is ready with "
                        f"address {remote_address}")

            ssh_hook = ssh_hooks.SSHHook(
                username=self._server_username, key_file=self._key_file,
                remote_host=remote_address, cmd_timeout=120, conn_timeout=120)
            self.ssh_hook = ssh_hook

            super().execute(context)
        finally:
            openstack.delete_server(conn, server.id)
