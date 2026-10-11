"""Lazy public exports; importing the package initializes no optional backend."""

from importlib import import_module

_EXPORTS = {
    "CORRUPT_MANIFEST": ("cache_manifest", "CORRUPT_MANIFEST"),
    "MISSING_ARTIFACT": ("cache_manifest", "MISSING_ARTIFACT"),
    "OK": ("cache_manifest", "OK"),
    "STALE": ("cache_manifest", "STALE"),
    "UNVERIFIED_NO_MANIFEST": ("cache_manifest", "UNVERIFIED_NO_MANIFEST"),
    "atomic_path": ("cache_manifest", "atomic_path"),
    "fingerprint_file": ("cache_manifest", "fingerprint_file"),
    "manifest_path_for": ("cache_manifest", "manifest_path_for"),
    "read_manifest": ("cache_manifest", "read_manifest"),
    "source_fingerprint": ("cache_manifest", "source_fingerprint"),
    "verify_artifact_set": ("cache_manifest", "verify_artifact_set"),
    "verify_manifest": ("cache_manifest", "verify_manifest"),
    "write_manifest": ("cache_manifest", "write_manifest"),
    "compute_config_hash": ("config", "compute_config_hash"),
    "load_yaml_config": ("config", "load_yaml_config"),
    "check_h5_immutability": ("h5_guard", "check_h5_immutability"),
    "is_immutable_artifact": ("h5_guard", "is_immutable_artifact"),
    "safe_open_h5": ("h5_guard", "safe_open_h5"),
    "compact_kpms_checkpoint": ("io", "compact_kpms_checkpoint"),
    "copy_dataset": ("file_io", "copy_dataset"),
    "load_npz_safe": ("file_io", "load_npz_safe"),
    "load_pickle_safe": ("file_io", "load_pickle_safe"),
    "save_npz_atomic": ("file_io", "save_npz_atomic"),
    "save_pickle_atomic": ("file_io", "save_pickle_atomic"),
    "get_logger": ("logging", "get_logger"),
    "send_ntfy": ("notify", "send_ntfy"),
    "ARCHIVE_DIR": ("paths", "ARCHIVE_DIR"),
    "CONFIGS_DIR": ("paths", "CONFIGS_DIR"),
    "DATA_ROOT": ("paths", "DATA_ROOT"),
    "DOCS_DIR": ("paths", "DOCS_DIR"),
    "OUTPUTS_DIR": ("paths", "OUTPUTS_DIR"),
    "REPO_ROOT": ("paths", "REPO_ROOT"),
    "TESTS_DIR": ("paths", "TESTS_DIR"),
    "ensure_dir": ("paths", "ensure_dir"),
    "get_data_root": ("paths", "get_data_root"),
    "get_repo_root": ("paths", "get_repo_root"),
    "resolve_data_path": ("paths", "resolve_data_path"),
    "resolve_path": ("paths", "resolve_path"),
    "to_relative_path": ("paths", "to_relative_path"),
    "RunRecord": ("run_record", "RunRecord"),
    "DORSAL_17KP_NAMES": ("fpcode.contracts.pose", "DORSAL_17KP_NAMES"),
    "FUSED_21KP_NAMES": ("fpcode.contracts.pose", "FUSED_21KP_NAMES"),
    "FUSED_23KP_NAMES": ("fpcode.contracts.pose", "FUSED_23KP_NAMES"),
    "VENTRAL_4KP_NAMES": ("fpcode.contracts.pose", "VENTRAL_4KP_NAMES"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        module = import_module(module_name) if module_name.startswith("fpcode.") else import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
