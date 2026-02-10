import os
import sys
import logging
import threading
from enum import Enum
from typing import Optional

import transformers


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"


_log_level_mapping = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
}


_RESET = "\x1b[0m"
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_CYAN = "\x1b[36m"


class ColoredMessageFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: _GREEN,
        logging.INFO: _CYAN,
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, _RESET)
        format_str = f"{color}%(asctime)s-%(name)s-%(levelname)s:{_RESET} %(message)s"
        formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")
        return formatter.format(record)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    """
    Configure the library root logger.
    Modified from [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/logging.py).
    """
    global _default_handler

    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        library_root_logger.propagate = False

        _default_handler.setFormatter(ColoredMessageFormatter())


def set_verbosity(verbosity: LogLevel) -> None:
    _configure_library_root_logger()
    verbosity = _log_level_mapping[verbosity]
    _get_library_root_logger().setLevel(verbosity)
    transformers.utils.logging.set_verbosity(verbosity)


def get_logger(name: str) -> logging.Logger:
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)
