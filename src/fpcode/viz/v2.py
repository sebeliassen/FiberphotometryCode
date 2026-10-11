"""Compatibility facade. Implementations live in focused modules.

Historical imports and object identities remain available; optional backends load
only when a backend attribute is requested.
"""

from importlib import import_module

_EXPORTS = {'EXP_DIR': 'fpcode.viz.legacy_defaults', 'SCRIPT_DIR': 'fpcode.viz.legacy_defaults', 'COHORT_DIR': 'fpcode.viz.legacy_defaults', 'MODEL_NAME': 'fpcode.viz.legacy_defaults', 'MODEL_DIR': 'fpcode.viz.legacy_defaults', 'RESULTS_PATH': 'fpcode.viz.legacy_defaults', 'CONFIG_PATH': 'fpcode.viz.legacy_defaults', 'VFI_PATH': 'fpcode.viz.legacy_defaults', 'PREPROCESSED_PATH': 'fpcode.viz.legacy_defaults', 'FUSED_DIR': 'fpcode.viz.legacy_defaults', 'TOP_VID_DIR': 'fpcode.viz.legacy_defaults', 'BOT_VID_DIR': 'fpcode.viz.legacy_defaults', 'VIZ_VERSION': 'fpcode.viz.legacy_defaults', 'DEFAULT_VIZ_DIR': 'fpcode.viz.legacy_defaults', 'DORSAL_BODYPARTS': 'fpcode.viz.legacy_defaults', 'PAW_BODYPARTS': 'fpcode.viz.legacy_defaults', 'BODYPARTS_21KP': 'fpcode.viz.legacy_defaults', 'DORSAL_IDXS': 'fpcode.viz.legacy_defaults', 'PAW_IDXS': 'fpcode.viz.legacy_defaults', 'FRONT_PAW_BODYPARTS': 'fpcode.viz.legacy_defaults', 'HIND_PAW_BODYPARTS': 'fpcode.viz.legacy_defaults', 'FRONT_PAW_IDXS': 'fpcode.viz.legacy_defaults', 'HIND_PAW_IDXS': 'fpcode.viz.legacy_defaults', 'BOT_NOSE': 'fpcode.viz.legacy_defaults', 'BOT_TAIL': 'fpcode.viz.legacy_defaults', 'BOT_PAW_NAMES': 'fpcode.viz.legacy_defaults', 'CHAMBER_TO_ARENA': 'fpcode.viz.legacy_defaults', 'SCHEMA_VERSION': 'fpcode.viz.legacy_defaults', 'DENSITY_PCA_DIM': 'fpcode.viz.legacy_defaults', 'DENSITY_N_NEIGHBORS': 'fpcode.viz.legacy_defaults', 'DENSITY_PCA_SAMPLES': 'fpcode.viz.legacy_defaults', 'KPMS_FILTER_SIZE': 'fpcode.viz.legacy_defaults', 'DEFAULT_SCORE_PRE_SECONDS': 'fpcode.viz.legacy_defaults', 'DEFAULT_SCORE_POST_SECONDS': 'fpcode.viz.legacy_defaults', 'DEFAULT_MOVIE_PRE_SECONDS': 'fpcode.viz.legacy_defaults', 'DEFAULT_MOVIE_POST_SECONDS': 'fpcode.viz.legacy_defaults', 'DEFAULT_MIN_BOUT_SECONDS': 'fpcode.viz.legacy_defaults', 'frames_from_seconds': 'fpcode.viz.selection.bouts', 'make_window': 'fpcode.viz.selection.bouts', 'half_open_bouts': 'fpcode.viz.selection.bouts', 'make_bout_id': 'fpcode.viz.selection.bouts', 'stable_ranks': 'fpcode.viz.selection.bouts', 'base_session_id': 'fpcode.viz.selection.bouts', 'get_session_files': 'fpcode.viz.legacy_io', 'filter_segment_pose': 'fpcode.viz.geometry', 'egocentric_window': 'fpcode.viz.geometry', 'movie_context_ok': 'fpcode.viz.selection.eligibility', 'require_fused_qc': 'fpcode.viz.selection.eligibility', 'active_bout_window': 'fpcode.viz.geometry', 'score_execution_counts': 'fpcode.viz.geometry', 'shared_active_rms': 'fpcode.viz.geometry', 'fusion_geometry': 'fpcode.viz.geometry', 'canonical_crop_matrices': 'fpcode.viz.geometry', 'source_input_hashes': 'fpcode.viz.provenance', 'verify_manifest_inputs': 'fpcode.viz.provenance', 'media_file_identity': 'fpcode.viz.legacy_io', 'selection_provenance_matches': 'fpcode.viz.provenance', 'wrap_angle_deg': 'fpcode.viz.geometry', 'QCConfig': 'fpcode.viz.selection.eligibility', 'valid_track_mask': 'fpcode.viz.selection.eligibility', 'longest_true_run': 'fpcode.viz.selection.eligibility', 'window_frac_gap': 'fpcode.viz.selection.eligibility', 'summarize_window_qc': 'fpcode.viz.selection.eligibility', 'summarize_frac_gap': 'fpcode.viz.selection.eligibility', 'qc_decision': 'fpcode.viz.selection.eligibility', 'collapse_qc_reason': 'fpcode.viz.selection.eligibility', 'load_fused_tracks': 'fpcode.viz.legacy_io', 'choose_tier': 'fpcode.viz.selection.density', 'density_ratio_scores': 'fpcode.viz.selection.density', 'order_group_exemplar_first': 'fpcode.viz.selection.density', 'select_tier1_density': 'fpcode.viz.selection.density', '_medoid_and_order': 'fpcode.viz.selection.density', 'select_tier2_compact': 'fpcode.viz.selection.density', 'select_tier3_all': 'fpcode.viz.selection.density', 'assert_unique_bout_ids': 'fpcode.viz.selection.exemplars', 'selection_ids': 'fpcode.viz.selection.exemplars', 'resolve_display_bouts': 'fpcode.viz.selection.exemplars', 'assert_rendered_equals_manifest': 'fpcode.viz.selection.exemplars', 'grid_layout': 'fpcode.viz.selection.exemplars', 'compute_robust_bot_pose': 'fpcode.viz.geometry', 'sha256_file': 'fpcode.viz.hashing', 'sha256_json': 'fpcode.viz.hashing', '_json_default': 'fpcode.viz.hashing', 'NumpyEncoder': 'fpcode.viz.hashing', 'HashCache': 'fpcode.viz.hashing', 'CALIBRATION_FILES': 'fpcode.viz.legacy_defaults', 'hash_files': 'fpcode.viz.hashing', 'manifest_syllable_hash': 'fpcode.viz.provenance', 'manifest_global_hash': 'fpcode.viz.provenance', 'make_cache_key': 'fpcode.viz.provenance', 'sidecar_path': 'fpcode.viz.provenance', 'write_sidecar': 'fpcode.viz.provenance', 'cache_decision': 'fpcode.viz.provenance', 'kpms_provenance': 'fpcode.viz.provenance', 'rms_point_difference': 'fpcode.viz.geometry', 'confidence_aware_rms': 'fpcode.viz.geometry', 'write_json_atomic': 'fpcode.viz.hashing'}

for _name in (
    "fusion_geometry", "compute_robust_bot_pose", "summarize_window_qc", "summarize_frac_gap",
    "source_input_hashes", "verify_manifest_inputs", "selection_provenance_matches", "make_cache_key",
):
    _EXPORTS[_name] = "fpcode.viz.legacy_api"

__all__ = [name for name in _EXPORTS if not name.startswith("_")]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
