"""Independent boundary tests and a historical-update identity supplement."""

import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.reference_seqnmf import backend, fit, infer
from studies.karen_pff_vs_monomer.analysis.run_reference_seqnmf import (
    evaluate,
    frame_bouts,
    merge_bouts,
    pack_chunks,
)


def test_rle_keeps_exclusions_and_missing_frames_as_barriers():
    actual = frame_bouts(np.array([10, 11, 12, 14, 15, 16, 17]),
                         np.array([1, 1, 2, 2, 2, 3, 3]), np.array([0, 0, 1, 0, 0, 0, 0], bool))
    assert actual[["token", "start_frame", "end_frame_exclusive"]].values.tolist() == [
        [1, 10, 12], [2, 14, 16], [3, 16, 18]]
    assert actual.block.iloc[0] != actual.block.iloc[1]
    mapping = np.array([0, 1, 1, 1])
    merged = merge_bouts(actual, mapping)
    assert merged[["token", "start_frame", "end_frame_exclusive"]].values.tolist() == [[1, 10, 12], [1, 14, 18]]


def chunk(tokens, first=0, chunk_id=0):
    n = len(tokens)
    return pd.DataFrame({"token": tokens, "start_frame": np.arange(first, first + n),
                         "end_frame_exclusive": np.arange(first + 1, first + n + 1),
                         "block": chunk_id, "chunk_id": chunk_id, "animal_id": "mouse",
                         "session_id": "recording", "condition": "3w_PI"})


def test_packed_onsets_cannot_use_padding_or_cross_chunks():
    x, valid, tokens, references = pack_chunks([chunk([0, 1, 2]), chunk([2, 1, 0], 10, 1)], np.arange(3), 3, 3)
    assert np.flatnonzero(valid).tolist() == [0, 6]
    assert tokens.tolist() == [0, 1, 2, -1, -1, -1, 2, 1, 0, -1, -1, -1]
    assert np.all(x[:, tokens == -1] == 0)
    assert references.iloc[0].start_frame == 0 and references.iloc[8].end_frame_exclusive == 13


def test_five_bout_windows_keep_short_chunks_and_recording_barriers_unavailable():
    x, valid, tokens, refs = pack_chunks(
        [chunk([0, 1, 2, 3], 0, 0), chunk([0, 1, 2, 3, 4, 0], 20, 1)], np.arange(5), 5, 5)
    assert np.flatnonzero(valid).tolist() == [9, 10]
    assert tokens[tokens >= 0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0]
    assert np.all(x[:, tokens < 0] == 0)
    assert refs.iloc[9].start_frame == 20 and refs.iloc[14].end_frame_exclusive == 26


def test_five_bout_partial_support_has_explicit_normalized_error_geometry():
    _, valid, tokens, refs = pack_chunks([chunk([4, 0, 1, 2, 4, 3])], np.arange(5), 5, 5)
    w = np.eye(5, dtype=np.float32)[:, None, :]
    h = np.zeros((1, len(tokens)), np.float32)
    h[0, 1] = 2
    metrics, calls = evaluate(w, h, tokens, valid, refs, np.array([1.]))
    assert len(calls) == 1
    assert calls.iloc[0].stage_support == pytest.approx(3 / 5)
    assert calls.iloc[0].template_error == pytest.approx(.64)
    assert calls.iloc[0].end_frame_exclusive == 6
    assert metrics["fit_union_time_coverage"] == pytest.approx(5 / 6)


def test_unsupported_activations_stay_exactly_zero():
    x, valid, _, _ = pack_chunks([chunk([0, 1, 2, 0]), chunk([1, 2, 0], 10, 1)], np.arange(3), 3, 3)
    runtime = backend("cpu")
    w, h, _ = fit(x, valid, factors=2, length=3, penalty=.3, iterations=5, seed=42, runtime=runtime)
    held = infer(w, x, valid, penalty=.3, iterations=5, seed=42, runtime=runtime)
    assert np.all(h[:, ~valid] == 0) and np.all(held[:, ~valid] == 0)
    assert np.isfinite(w).all() and np.isfinite(h).all()


def test_fixed_template_inference_is_independent_across_separated_blocks():
    runtime = backend("cpu")
    w = np.random.RandomState(1).uniform(.1, 1., (3, 2, 3)).astype(np.float32)
    first = pack_chunks([chunk([0, 1, 2, 0, 1])], np.arange(3), 3, 3)
    second = pack_chunks([chunk([2, 2, 0, 1, 0], 20, 1)], np.arange(3), 3, 3)
    # Equal deterministic initial H isolates the computational boundary check.
    from studies.karen_pff_vs_monomer.analysis.reference_seqnmf import update
    a, b = first[0], second[0]
    va, vb = first[1], second[1]
    h = np.full((2, a.shape[1] + b.shape[1]), .5, np.float32)
    v = np.r_[va, vb]
    h[:, ~v] = 0
    _, joint = update(w, h, np.concatenate([a, b], axis=1), v, .3, templates=False)
    _, separate = update(w, h[:, :a.shape[1]], a, va, .3, templates=False)
    np.testing.assert_allclose(joint[:, :a.shape[1]], separate, atol=1e-7)
    assert runtime[2]["device"] == "cpu"


def test_adapted_unmasked_update_matches_historical_cpu_fit():
    root = Path(__file__).resolve().parents[2]
    source = root / "studies/sequence_viz/optimized_seqnmf/convolutional_seqnmf.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in {"shift_right_np", "shift_left_np", "ConvolutionalSeqNMF"}]
    scope = {"np": np, "time": time, "HAS_JAX_GPU": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    x = np.eye(3, dtype=np.float32)[:, [0, 1, 2, 1, 0, 2, 0, 1, 2]]
    old = scope["ConvolutionalSeqNMF"](K=2, L=3, lambda_ortho=.3, max_iter=3, random_seed=42, use_gpu=False).fit(x)
    w, h, _ = fit(x, np.ones(x.shape[1], bool), factors=2, length=3, penalty=.3,
                   iterations=3, seed=42, runtime=backend("cpu"))
    np.testing.assert_allclose(w, old.W, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(h, old.H, atol=2e-6, rtol=2e-6)


def test_detected_end_is_last_included_bout_and_gaps_are_rejected():
    _, valid, tokens, refs = pack_chunks([chunk([0, 1, 2, 0, 1, 2])], np.arange(3), 3, 3)
    w = np.eye(3, dtype=np.float32)[:, None, :]
    h = np.zeros((1, len(tokens)), np.float32)
    h[0, 1] = 2
    # At onset 1 the supported observed sequence is [1,2,0].
    w = w[[2, 0, 1]]
    metrics, calls = evaluate(w, h, tokens, valid, refs, np.array([1.]))
    assert len(calls) == 1
    assert calls.iloc[0].start_frame == 1 and calls.iloc[0].end_frame_exclusive == 4
    assert calls.iloc[0].duration_sec == pytest.approx(3 / 60)
    assert metrics["fit_union_time_coverage"] == pytest.approx(.5)
    refs.loc[2, "start_frame"] += 1
    with pytest.raises(AssertionError, match="excluded/native"):
        evaluate(w, h, tokens, valid, refs, np.array([1.]))
