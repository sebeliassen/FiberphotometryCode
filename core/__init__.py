"""
Core foundational library for FiberphotometryCode.
Single Source of Truth (SSOT) for paths, schemas, immutability guards,
execution records, and structured logging.
"""

from .h5_guard import check_h5_immutability, safe_open_h5
from .logging_utils import get_logger, send_ntfy
from .paths import get_repo_root, resolve_path, to_relative_path
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
