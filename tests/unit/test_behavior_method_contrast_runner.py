"""Bounded runner policy checks plus a small end-to-end categorical fixture."""

import json

import numpy as np
import pandas as pd
import pytest

from fpcode.behavior.unit_adapters import from_frame_labels
from fpcode.contracts import SessionKey
from studies.sequence_viz.run_method_contrasts import (
    load_manifest,
    main,
    propose_exact_motifs,
    run_pilot,
    select_native_window,
)


def _manifest(tmp_path):
    sessions = []
    for animal in ("train_a", "train_b", "held"):
        path = tmp_path/f"{animal}.npz"
        # Repeated heterogeneous routines plus a physical missing-frame barrier.
        tokens = np.tile([0, 1, 2, 3, 0, 2, 1, 3], 8)
        labels = np.repeat(tokens, 2)
        frames = np.arange(len(labels))
        frames[len(frames)//2:] += 5
        np.savez(path, native_frames=frames, labels=labels)
        sessions.append(dict(recording=animal, subject=animal, npz_path=str(path), native_mapping_verified=True,
                             native_mapping_evidence="fixture integer native frames with explicit five-frame gap"))
    manifest = dict(schema_version=1, cohorts=[dict(cohort_id="fixture", model_id="fixture_kpms", clock="video", fps=10.,
                     vocabulary=["0", "1", "2", "3"], train_subjects=["train_a", "train_b"], held_out_subjects=["held"], sessions=sessions)])
    path = tmp_path/"pilot_data.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_rejects_split_leakage_and_unverified_native_mapping(tmp_path):
    path, manifest = _manifest(tmp_path)
    manifest["cohorts"][0]["held_out_subjects"] = ["train_a"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint"):
        load_manifest(path)
    path, manifest = _manifest(tmp_path)
    manifest["cohorts"][0]["sessions"][0]["native_mapping_verified"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verified evidence"):
        load_manifest(path)
    path, manifest = _manifest(tmp_path)
    manifest["cohorts"][0]["cohort_id"] = ".."
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filesystem-safe"):
        load_manifest(path)


def test_whole_bout_windows_preserve_native_gaps_and_original_intervals():
    frames = np.r_[np.arange(12), np.arange(20, 32)]
    labels = np.tile(np.repeat([0, 1, 2], 2), 4)
    stream = from_frame_labels(frames, labels, session=SessionKey("s", "a"), model_id="model", clock="video", fps=10.)
    selected = select_native_window(stream, event_limit=6, rng=np.random.default_rng(1))
    assert len(selected.occurrences) == 6
    assert set(selected.occurrences) <= set(stream.occurrences)
    assert selected.valid_intervals[0].stop-selected.valid_intervals[0].start == 12
    assert selected.valid_intervals[0].start in (0, 20)
    motifs, evidence = propose_exact_motifs([selected], min_prevalence=1)
    assert motifs and {role for row in evidence for role in row["roles"]} == {"frequent", "diverse"}
    assert len(set(motifs.values())) == len(motifs)


def test_metadata_only_cli_creates_no_outputs(tmp_path, capsys):
    manifest, _ = _manifest(tmp_path)
    output = tmp_path/"output"
    main(["--manifest", str(manifest), "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["execute"] is False
    assert not output.exists()


def test_window_selection_uses_requested_count_or_longest_available_block():
    frames = np.r_[np.arange(8), np.arange(20, 60)]
    labels = np.r_[np.tile([0, 1], 4), np.tile([0, 1], 20)]
    stream = from_frame_labels(frames, labels, session=SessionKey("s", "a"), model_id="m", clock="video", fps=10.)
    for limit, expected in ((24, 24), (50, 40)):
        selected = select_native_window(stream, event_limit=limit, rng=np.random.default_rng(1))
        assert len(selected.occurrences) == expected
        assert selected.valid_intervals[0].start >= 20


def test_small_categorical_pilot_exports_all_methods_and_truthful_effort(tmp_path):
    manifest, _ = _manifest(tmp_path)
    output = tmp_path/"output"
    effort = run_pilot(manifest, output, train_event_limit=24, held_event_limit=24, nmf_iterations=2, hsmm_iterations=1)
    assert (output/"effort.json").is_file()
    windows = pd.read_csv(output/"matched_windows.csv")
    assert len(windows) == 3 and windows.source_events.max() <= 24
    assert set(windows.loc[windows.split == "train", "subject"]) == {"train_a", "train_b"}
    calls = pd.read_csv(output/"calls.csv")
    assert set(calls.subject) <= {"held"}
    panels = pd.read_csv(output/"catalogues_quality.csv")
    assert len(panels) == 8  # Exact + four seqNMF fits + configured + two learned HSMM fits.
    fitted = [row for row in effort["conditions"] if row.get("method", "").startswith("hsmm_fitted")]
    assert len(fitted) == 2
    assert all(row["fit_diagnostics"]["initialization"] == "dirichlet" for row in fitted)
    assert all(row["training_event_counts_by_animal"] == {"train_a": 24, "train_b": 24} for row in fitted)
    with np.load(output/"fixture"/"hsmm_configured.npz", allow_pickle=False) as configured:
        durations = configured["log_duration_pmfs"]
    for seed in (101, 202):
        with np.load(output/"fixture"/f"hsmm_fitted_seed{seed}.npz", allow_pickle=False) as fitted_model:
            np.testing.assert_array_equal(fitted_model["log_duration_pmfs"], durations)
    assert "quality_score" not in panels and "winner" not in effort
