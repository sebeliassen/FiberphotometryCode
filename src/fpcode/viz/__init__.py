"""
fpcode.viz
==========
Visualization tools for posture tracking, syllable grid movies,
density-sampled trajectory renders, and quality control.
"""

from .v2 import (
    HashCache,
    QCConfig,
    active_bout_window,
    canonical_crop_matrices,
    choose_tier,
    compute_robust_bot_pose,
    density_ratio_scores,
    egocentric_window,
    filter_segment_pose,
    frames_from_seconds,
    grid_layout,
    half_open_bouts,
    make_bout_id,
    make_window,
    movie_context_ok,
    qc_decision,
    select_tier1_density,
    select_tier2_compact,
    select_tier3_all,
    sha256_file,
    sha256_json,
    shared_active_rms,
    stable_ranks,
)

__all__ = [
    "frames_from_seconds",
    "make_window",
    "half_open_bouts",
    "make_bout_id",
    "stable_ranks",
    "filter_segment_pose",
    "egocentric_window",
    "movie_context_ok",
    "active_bout_window",
    "shared_active_rms",
    "canonical_crop_matrices",
    "QCConfig",
    "qc_decision",
    "choose_tier",
    "density_ratio_scores",
    "select_tier1_density",
    "select_tier2_compact",
    "select_tier3_all",
    "grid_layout",
    "compute_robust_bot_pose",
    "sha256_file",
    "sha256_json",
    "HashCache",
]
