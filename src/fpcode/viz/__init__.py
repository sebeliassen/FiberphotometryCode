"""Lazy public exports; importing the package initializes no optional backend."""

from importlib import import_module

_EXPORTS = {
    "plot_syllable_usage": ("figures", "plot_syllable_usage"),
    "plot_transition_matrix": ("figures", "plot_transition_matrix"),
    "generate_grid_movies": ("grid_movies", "generate_grid_movies"),
    "plot_syllable_trajectories": ("trajectories", "plot_syllable_trajectories"),
    "HashCache": ("v2", "HashCache"),
    "QCConfig": ("v2", "QCConfig"),
    "active_bout_window": ("v2", "active_bout_window"),
    "canonical_crop_matrices": ("v2", "canonical_crop_matrices"),
    "choose_tier": ("v2", "choose_tier"),
    "compute_robust_bot_pose": ("v2", "compute_robust_bot_pose"),
    "density_ratio_scores": ("v2", "density_ratio_scores"),
    "egocentric_window": ("v2", "egocentric_window"),
    "filter_segment_pose": ("v2", "filter_segment_pose"),
    "frames_from_seconds": ("v2", "frames_from_seconds"),
    "grid_layout": ("v2", "grid_layout"),
    "half_open_bouts": ("v2", "half_open_bouts"),
    "make_bout_id": ("v2", "make_bout_id"),
    "make_window": ("v2", "make_window"),
    "movie_context_ok": ("v2", "movie_context_ok"),
    "qc_decision": ("v2", "qc_decision"),
    "select_tier1_density": ("v2", "select_tier1_density"),
    "select_tier2_compact": ("v2", "select_tier2_compact"),
    "select_tier3_all": ("v2", "select_tier3_all"),
    "sha256_file": ("v2", "sha256_file"),
    "sha256_json": ("v2", "sha256_json"),
    "shared_active_rms": ("v2", "shared_active_rms"),
    "stable_ranks": ("v2", "stable_ranks"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(f".{module_name}", __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
