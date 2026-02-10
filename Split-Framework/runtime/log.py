"""Runtime logging utilities.

What this file is for:
- Provide a single `Log` helper used throughout the framework.
- Standardize log formatting and ensure per-rank log files are created.

If you need logging, import `Log` from here.
"""

from __future__ import annotations

import abc
import logging
import os
import time
import sys


def get_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class AbstractLog(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def info(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def error(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def debug(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def warning(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def critical(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError


class Log(AbstractLog):
    """Simple file logger used across the project."""

    def __init__(self, className, parse):
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(process)d \n\t %(message)s",
            )

        self.className = className
        raw_path = parse["log_save_path"]

        try:
            rank = parse["rank"]
        except Exception:
            rank = None
        if rank is None:
            rank = "unknown"
        try:
            self.savePath = str(raw_path).format(rank=rank, pid=os.getpid(), className=className)
        except Exception:
            self.savePath = str(raw_path)

        # Ensure the output directory exists.
        log_dir = os.path.dirname(self.savePath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(className)
        self.logger.setLevel(level=logging.INFO)
        # Prevent duplicate output via root logger handlers.
        self.logger.propagate = False

        # IMPORTANT: one experiment should produce ONE log file.
        # Allow all ranks to write to the same log file so client-side metrics are captured.

        abs_save_path = os.path.abspath(self.savePath)
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == abs_save_path:
                self.handler = handler
                break
        else:
            self.handler = logging.FileHandler(self.savePath)
            self.logger.addHandler(self.handler)

        # Also log to console so users can watch progress live.
        has_stream = any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers)
        if not has_stream:
            self.stream_handler = logging.StreamHandler(stream=sys.stdout)
            self.logger.addHandler(self.stream_handler)
        else:
            self.stream_handler = None

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s")
        self.handler.setFormatter(formatter)
        if self.stream_handler is not None:
            self.stream_handler.setFormatter(formatter)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.info(message)

    def critical(self, message):
        self.logger.info(message)

    def save(self):
        return
