"""Native-clock frozen projection and first-order controls have independent oracles."""

import base64

import numpy as np
import pandas as pd
import pytest

from fpcode.statistics.chain_nulls import sample_first_order_bouts
from fpcode.workflows.sequence_search import frozen_seqnmf_model, project_frozen_catalogue, score_frozen_sequence_batch


def bouts(tokens, starts=None):
    starts = np.arange(len(tokens)) * 10 if starts is None else np.asarray(starts)
    return pd.DataFrame({"syllable_id": tokens, "segment": 0, "start_frame": starts,
                         "end_frame_exclusive": starts + 10, "duration_s": .5})


def config(method="exact"):
    return {"method": method, "min_bout_ms": 0, "n_units": 1, "chain_len": 3, "penalty": 0.}


def artifact():
    w = np.eye(3, dtype="<f4")[:, None, :]
    return {"templates": {"dtype": "<f4", "encoding": "base64", "shape": list(w.shape),
                           "data": base64.b64encode(w.tobytes()).decode()}}


def test_exact_projection_retains_bounds_and_does_not_join_native_gaps():
    source = bouts([0, 1, 2, 0, 1, 2], [0, 10, 20, 40, 50, 70])
    original = source.copy(deep=True)
    events, catalog = project_frozen_catalogue(source, {"motifs": [[0, 1, 2]]}, config(), vocabulary=3, fps=20)
    assert events.start_frame.tolist() == [0]
    assert events.end_frame_exclusive.tolist() == [30]
    assert events.duration_s.tolist() == [1.5]
    assert catalog.tokens.tolist() == ["0 1 2"]
    pd.testing.assert_frame_equal(source, original)


def test_decoder_checks_shape_bytes_and_no_fit(monkeypatch):
    from fpcode.behavior.episodes.seqnmf import SeqNMF

    monkeypatch.setattr(SeqNMF, "fit", lambda *a, **k: pytest.fail("projection must not fit"))
    saved = artifact()
    model = frozen_seqnmf_model(saved, config("seqnmf"), vocabulary=3)
    np.testing.assert_array_equal(model.w[:, 0], np.eye(3))
    with pytest.raises(ValueError, match="shape"):
        frozen_seqnmf_model(saved, config("seqnmf"), vocabulary=4)
    saved["templates"]["data"] = base64.b64encode(b"x").decode()
    with pytest.raises(ValueError, match="byte count"):
        frozen_seqnmf_model(saved, config("seqnmf"), vocabulary=3)


def test_seqnmf_projection_is_frozen_and_requires_external_thresholds():
    source = bouts([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="thresholds"):
        project_frozen_catalogue(source, artifact(), config("seqnmf"), vocabulary=3, fps=20)
    events, _ = project_frozen_catalogue(source, artifact(), config("seqnmf"), vocabulary=3, fps=20, thresholds=[.1])
    assert events.start_frame.tolist() == [0, 30]
    assert events.end_frame_exclusive.tolist() == [30, 60]
    assert (events.template_error == 0).all()


def test_first_order_null_preserves_structural_cycle_and_segment_boundaries():
    refs = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    out = sample_first_order_bouts(refs, length=12, initial=0, vocabulary=3, replicates=7, seed=2)
    # State 2 has no observed successor, so its fallback supports 0 and 1;
    # the segment boundary must not introduce a deterministic 2->0 transition.
    assert set(out[:, 3]) == {0, 1}
    assert np.all(out[:, 1] == 1) and np.all(out[:, 2] == 2)
    assert np.all(out[:, 1:] != out[:, :-1])
    np.testing.assert_array_equal(out, sample_first_order_bouts(refs, length=12, initial=0, vocabulary=3, replicates=7, seed=2))


def test_first_order_null_respects_budget_and_rejects_non_rle_reference():
    with pytest.raises(MemoryError):
        sample_first_order_bouts([np.array([0, 1])], length=1000000, initial=0, vocabulary=2,
                                 replicates=100, seed=1, memory_mb=1)
    with pytest.raises(ValueError, match="adjacent identical"):
        sample_first_order_bouts([np.array([0, 0])], length=3, initial=0, vocabulary=2, replicates=2, seed=1)


def test_batched_null_inference_matches_separated_timelines():
    samples = {"routine": [np.tile([0, 1, 2], 5), np.array([0, 1, 2])],
               "other": [np.tile([0, 2, 1], 6)]}
    cfg = config("seqnmf")
    batch = score_frozen_sequence_batch(samples, artifact(), cfg, vocabulary=3)
    for key, sample in samples.items():
        independent = score_frozen_sequence_batch({key: sample}, artifact(), cfg, vocabulary=3)
        assert batch[key] == pytest.approx(independent[key], abs=1e-6)
    assert batch["routine"] < batch["other"]
    with pytest.raises(MemoryError):
        score_frozen_sequence_batch({"big": [np.tile([0, 1, 2], 10000)]}, artifact(), cfg, vocabulary=3, memory_mb=1)
