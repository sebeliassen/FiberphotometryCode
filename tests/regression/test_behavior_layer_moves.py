"""Old behavioural imports retain the exact canonical callable identities.

Independent numerical expectations remain in the behavioural unit tests. These
checks protect the import/pickle lookup paths and private helpers during moves.
"""

import importlib

import pytest

MOVED_APIS = (
    (
        "fpcode.behavior.kinematics",
        "fpcode.behavior.observations.kinematics",
        ("movement_changepoint_score", "detect_movement_initiations",
         "legacy_nose_tail_kinematics", "smooth_heading_circular",
         "smooth_centroid_median", "clean_short_runs", "presence_mask_from_conf",
         "subset_by_activity", "smooth_heading_circular_legacy_fusion",
         "smooth_centroid_median_legacy_fusion", "presence_mask_from_conf_kpms"),
    ),
    (
        "fpcode.behavior.bouts",
        "fpcode.behavior.primitives.bouts",
        ("run_boundaries", "_contiguous_rows", "extract_bouts", "extract_syllable_runs"),
    ),
    (
        "fpcode.behavior.resolutions",
        "fpcode.behavior.primitives.resolutions",
        ("validate_state_map", "remap_states", "member_composition",
         "standardised_member_execution"),
    ),
    (
        "fpcode.behavior.sequences.counting",
        "fpcode.behavior.primitives.counting",
        ("label_counts",),
    ),
    (
        "fpcode.behavior.sequences.chains",
        "fpcode.behavior.episodes.chains",
        ("segmented_tokens", "ngram_counts", "edit_alignment", "exact_occurrences",
         "prefix_progression", "group_occurrences"),
    ),
    (
        "fpcode.behavior.sequences.seqnmf",
        "fpcode.behavior.episodes.seqnmf",
        ("reconstruct", "_adjoint", "_smooth", "SeqNMF", "template_similarity",
         "activation_events"),
    ),
    (
        "fpcode.behavior.sequences.organisation",
        "fpcode.behavior.analysis.organisation",
        ("valid_frame_intervals", "exclusive_timeline", "organisation_arrays",
         "eligible_pairs", "pair_audit", "return_masks", "repetition_runs",
         "organisation_metrics", "metric_values"),
    ),
    (
        "fpcode.behavior.transitions",
        "fpcode.behavior.analysis.transitions",
        ("adjacent_pairs", "compute_transition_matrix", "compute_stationary_distribution"),
    ),
    (
        "fpcode.behavior.variants",
        "fpcode.behavior.analysis.variants",
        ("fit_execution_variants", "assign_execution_variants"),
    ),
    (
        "fpcode.behavior.phenotyping",
        "fpcode.behavior.analysis.phenotyping",
        ("diversity", "frame_kinematics", "_finite_stat", "analyze_frames"),
    ),
    (
        "fpcode.behavior.sequences.prediction",
        "fpcode.behavior.analysis.prediction",
        ("HistoryPredictor",),
    ),
)


@pytest.mark.parametrize("old_name,canonical_name,names", MOVED_APIS)
def test_historical_callable_paths_resolve_to_canonical_implementation(
    old_name, canonical_name, names,
):
    historical = importlib.import_module(old_name)
    canonical = importlib.import_module(canonical_name)
    for name in names:
        original_path = getattr(historical, name)
        implementation = getattr(canonical, name)
        assert original_path is implementation
        assert implementation.__module__ == canonical_name
        if hasattr(implementation, "__globals__"):
            # A facade does not wrap or duplicate functions with different globals.
            assert implementation.__globals__ is vars(canonical)


def test_existing_package_exports_keep_canonical_callables():
    from fpcode import behavior
    from fpcode.behavior.analysis.transitions import compute_transition_matrix
    from fpcode.behavior.observations.kinematics import smooth_heading_circular
    from fpcode.behavior.primitives.bouts import extract_syllable_runs
    from fpcode.behavior.syllables import extract_syllable_runs as historical_runs

    assert behavior.compute_transition_matrix is compute_transition_matrix
    assert behavior.smooth_heading_circular is smooth_heading_circular
    assert behavior.extract_syllable_runs is extract_syllable_runs
    assert historical_runs is extract_syllable_runs
