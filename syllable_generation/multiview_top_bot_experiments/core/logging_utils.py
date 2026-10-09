"""
core.logging_utils
==================
Standardized structured logging and optional alert dispatching.
"""

import os
import sys
import logging
import urllib.request
from typing import Optional
from datetime import datetime

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def get_logger(name: str = "multiview_kpms", level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger with standard stream formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

def notify_status(
    title: str,
    msg: str,
    priority: str = "default",
    tags: str = "robot",
    enable_ntfy: bool = False,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Logs status and optionally dispatches a push notification via ntfy.
    Notification is disabled by default to prevent unwanted notification spam.
    """
    log = logger or get_logger()
    log.info(f"NOTIFICATION: {title} | {msg}")

    if not enable_ntfy:
        return True

    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": priority,
                "Tags": tags,
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception as e:
        log.warning(f"ntfy dispatch failed: {e}")
        return False
