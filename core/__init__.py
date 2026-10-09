"""
Core foundational library for FiberphotometryCode.
Single Source of Truth (SSOT) for paths, schemas, immutability guards,
execution records, and structured logging.
"""

from .paths import get_repo_root, to_relative_path, resolve_path
from .logging_utils import get_logger, send_ntfy
from .h5_guard import safe_open_h5, check_h5_immutability
from .run_record import RunRecord

__all__ = [
    "get_repo_root",
    "to_relative_path",
    "resolve_path",
    "get_logger",
    "send_ntfy",
    "safe_open_h5",
    "check_h5_immutability",
    "RunRecord",
]
