import datetime
import os
from logging import config, getLogger

logger_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"}},
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "fileHandler": {"class": "logging.FileHandler", "level": "WARN", "formatter": "simple", "filename": "finance_process"},
    },
    "loggers": {
        "finance_process": {"level": "WARN", "handlers": ["fileHandler"], "propagate": False},
        "finance_process.test": {"level": "DEBUG", "handlers": ["consoleHandler", "fileHandler"], "propagate": False},
    },
    "root": {"level": "DEBUG"},
}


def __init_logger():
    try:
        log_folder = os.path.join(os.path.dirname(__file__), "logs")
        if os.path.exists(log_folder) is False:
            os.makedirs(log_folder)
        log_file_path = f'{log_folder}/finance_process_{datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")}.log'
        logger_config["handlers"]["fileHandler"]["filename"] = log_file_path
        config.dictConfig(logger_config)

        if "FC_DEBUG" in os.environ:
            __DEBUG = bool(os.environ["FC_DEBUG"])
        else:
            __DEBUG = False

        if __DEBUG:
            return getLogger("finance_process.test")
        else:
            return getLogger("finance_process")
    except Exception as e:
        print(f"fail to set configure file: {e}")
        raise getLogger("finance_process")


__logger = __init_logger()


def info(message: str, **kwargs):
    try:
        __logger.info(message, **kwargs)
    except Exception:
        print(message)


def warning(message: str, **kwargs):
    try:
        __logger.warning(message, **kwargs)
    except Exception:
        print(message)


def warn(message: str, **kwargs):
    warning(message, **kwargs)


def error(message: str, **kwargs):
    try:
        __logger.error(message, **kwargs)
    except Exception:
        print(message)


def exception(message: str, **kwargs):
    try:
        __logger.exception(message, **kwargs)
    except Exception:
        print(message)


def critical(message, **kwargs):
    try:
        __logger.critical(message, **kwargs)
    except Exception:
        print(message)


def debug(message: str, **kwargs):
    try:
        __logger.debug(message, **kwargs)
    except Exception:
        print(message)
