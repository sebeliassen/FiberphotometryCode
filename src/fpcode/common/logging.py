"""
fpcode.common.logging
=====================
Standardized structured logging with console and file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def get_logger(
    name: str = "fiberphotometry",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Creates or retrieves a structured logger with standardized formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        existing_file_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == file_path.resolve()
        ]
        if not existing_file_handlers:
            file_formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger
