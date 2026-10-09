"""
fpcode.common
=============
Common foundational library: paths, config, logging, notifications,
HDF5 immutability guards, execution records, cache manifests, schemas, and IO.
"""

from .cache_manifest import (
    CORRUPT_MANIFEST,
    MISSING_ARTIFACT,
    OK,
    STALE,
    UNVERIFIED_NO_MANIFEST,
    atomic_path,
    fingerprint_file,
    manifest_path_for,
    read_manifest,
    source_fingerprint,
    verify_artifact_set,
    verify_manifest,
    write_manifest,
)
from .config import compute_config_hash, load_yaml_config
from .h5_guard import check_h5_immutability, is_immutable_artifact, safe_open_h5
from .io import (
    compact_kpms_checkpoint,
    copy_dataset,
    load_npz_safe,
    load_pickle_safe,
    save_npz_atomic,
    save_pickle_atomic,
)
from .logging import get_logger
from .notify import send_ntfy
from .paths import (
    ARCHIVE_DIR,
    CONFIGS_DIR,
    DATA_ROOT,
    DOCS_DIR,
    OUTPUTS_DIR,
    REPO_ROOT,
    TESTS_DIR,
    ensure_dir,
    get_data_root,
    get_repo_root,
    resolve_data_path,
    resolve_path,
    to_relative_path,
)
from .run_record import RunRecord
from .schemas import (
    DORSAL_17KP_NAMES,
    FUSED_21KP_NAMES,
    FUSED_23KP_NAMES,
    VENTRAL_4KP_NAMES,
)

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "CONFIGS_DIR",
    "DOCS_DIR",
    "TESTS_DIR",
    "OUTPUTS_DIR",
    "ARCHIVE_DIR",
    "get_repo_root",
    "get_data_root",
    "resolve_path",
    "resolve_data_path",
    "to_relative_path",
    "ensure_dir",
    "load_yaml_config",
    "compute_config_hash",
    "get_logger",
    "send_ntfy",
    "safe_open_h5",
    "check_h5_immutability",
    "is_immutable_artifact",
    "RunRecord",
    "atomic_path",
    "manifest_path_for",
    "write_manifest",
    "read_manifest",
    "verify_manifest",
    "verify_artifact_set",
    "fingerprint_file",
    "source_fingerprint",
    "OK",
    "UNVERIFIED_NO_MANIFEST",
    "MISSING_ARTIFACT",
    "STALE",
    "CORRUPT_MANIFEST",
    "DORSAL_17KP_NAMES",
    "VENTRAL_4KP_NAMES",
    "FUSED_21KP_NAMES",
    "FUSED_23KP_NAMES",
    "load_pickle_safe",
    "save_pickle_atomic",
    "load_npz_safe",
    "save_npz_atomic",
    "copy_dataset",
    "compact_kpms_checkpoint",
]
