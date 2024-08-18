from functools import cached_property
import logging

import paramiko

from airflow.providers.ssh.hooks.ssh import SSHHook


logger = logging.getLogger(__name__)


class SSHJumpHook(SSHHook):
    @cached_property
    def host_proxy(self) -> paramiko.Channel:

        conn = self.get_connection(self.ssh_conn_id)
        proxy_host = conn.extra_dejson.get("proxy_host")
        if not proxy_host:
            return

        proxy_port = conn.extra_dejson.get("proxy_port", 22)

        username = conn.login
        password = conn.password
        target_host = conn.host
        target_port = conn.port or 22

        logger.info(
            f"Using SSH proxy jump: {username}@{proxy_host}:{proxy_port}, "
            f"connecting to target: {username}@{target_host}:{target_port}")

        jumpbox = paramiko.SSHClient()
        jumpbox.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jumpbox.connect(proxy_host, username=username, password=password)

        jumpbox_transport = jumpbox.get_transport()
        src_addr = (proxy_host, proxy_port)
        dest_addr = (target_host, target_port)
        jumpbox_channel = jumpbox_transport.open_channel(
            "direct-tcpip", dest_addr, src_addr)
        return jumpbox_channel
