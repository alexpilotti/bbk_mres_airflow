import logging

from airflow import exceptions
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from airflow.contrib.hooks.sftp_hook import SFTPHook


logger = logging.getLogger(__name__)


class SFTPComparePathDatetimesSensor(BaseSensorOperator):
    @apply_defaults
    def __init__(self, path1, path2, sftp_hook,
                 *args, **kwargs):
        super(SFTPComparePathDatetimesSensor, self).__init__(*args, **kwargs)
        self.path1 = path1
        self.path2 = path2
        self.sftp_hook = sftp_hook

    def poke(self, context):
        if not self.sftp_hook.path_exists(self.path1):
            raise exceptions.AirflowFailException(
                f"Path does not exist: {self.path1}")
        if not self.sftp_hook.path_exists(self.path2):
            logger.info(f"Path does not exist: {self.path2}")
            return True

        t1 = self.sftp_hook.get_mod_time(self.path1)
        t2 = self.sftp_hook.get_mod_time(self.path2)
        logger.debug(f"t1: {t1} t2: {t2}")

        if t1 <= t2:
            raise exceptions.AirflowSkipException(
                f"{self.path2} is more recent than {self.path1}")

        return True
