import logging
import os

from airflow import exceptions
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults


logger = logging.getLogger(__name__)


class ComparePathDatetimesSensor(BaseSensorOperator):
    @apply_defaults
    def __init__(self, path1, path2, *args, **kwargs):
        super(ComparePathDatetimesSensor, self).__init__(*args, **kwargs)

        self.path1 = [path1] if isinstance(path1, str) else path1
        self.path2 = [path2] if isinstance(path2, str) else path2

    def poke(self, context):
        max_path1_time = None
        path1 = None
        for path in self.path1:
            if not os.path.exists(path):
                raise exceptions.AirflowFailException(
                    f"Path does not exist: {path}")

            t = os.path.getmtime(path)
            logger.debug(f"Mod time of \"{path}\": {t}")

            if not max_path1_time or t > max_path1_time:
                max_path1_time = t
                path1 = path

        path2 = None
        min_path2_time = None
        for path in self.path2:
            if not os.path.exists(path):
                logger.info(f"Path does not exist: {path}")
                return True

            t = os.path.getmtime(path)
            logger.debug(f"Mod time of \"{path}\": {t}")

            if not min_path2_time or t < min_path2_time:
                min_path2_time = t
                path2 = path

        if not min_path2_time:
            return True

        if max_path1_time <= min_path2_time:
            raise exceptions.AirflowSkipException(
                f"{path2} is more recent than {path1}")

        return True
