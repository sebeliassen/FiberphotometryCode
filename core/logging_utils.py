"""
core.logging_utils
==================
Centralized logging and notifications for FiberphotometryCode.
Provides structured logging with console + file handlers, and a unified
ntfy.sh notification helper with graceful network failure handling.
"""

import os
import sys
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Union

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
DEFAULT_NTFY_TOPIC = "woody_videos_seb"


def get_logger(
    name: str = "fiberphotometry",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates or retrieves a structured logger with standardized timestamp formatting.
    Optionally attaches a FileHandler if log_file is provided.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
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
        # Check if file handler already exists for this path
        existing_file_handlers = [
            h for h in logger.handlers
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


def send_ntfy(
    title: str,
    msg: str,
    priority: str = "default",
    tags: str = "robot",
    topic: Optional[str] = None,
) -> bool:
    """
    Sends notification to user's ntfy.sh endpoint.
    Configurable via NTFY_SERVER and NTFY_TOPIC environment variables.
    Fails gracefully if offline or misconfigured without interrupting pipeline execution.
    """
    # Check if disabled via environment
    if os.environ.get("NTFY_DISABLE", "0") in ("1", "true", "True"):
        return False

    server = os.environ.get("NTFY_SERVER", DEFAULT_NTFY_SERVER).rstrip("/")
    resolved_topic = topic or os.environ.get("NTFY_TOPIC", DEFAULT_NTFY_TOPIC).strip("/")
    url = f"{server}/{resolved_topic}"

    try:
        req = urllib.request.Request(
            url,
            data=msg.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": str(priority),
                "Tags": str(tags),
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            pass
        return True
    except Exception as e:
        # Graceful degradation - pipeline must continue
        sys.stderr.write(f"[ntfy warning] Notification '{title}' failed ({e})\n")
        sys.stderr.flush()
        return False
