"""Centralised rotating logger - เก็บใน ./data/logs/"""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from .paths import log_path

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "sweephunter") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_path(f"{name}.log"), maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    _LOGGERS[name] = logger
    return logger
