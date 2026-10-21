"""Independent synthetic checks for routine screening and native continuity."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.workflows.sequence_search import (
    activation_dead_fraction,
    evaluate_configuration,
    filter_short_bouts,
    motif_hits,
    motif_masks,
)


def _bouts(tokens, starts=None, lengths=None):
    tokens = np.asarray(tokens, dtype=int)
    lengths = np.full(len(tokens), 10, dtype=int) if lengths is None else np.asarray(lengths, dtype=int)
    starts = np.r_[0, np.cumsum(lengths)[:-1]] if starts is None else np.asarray(starts, dtype=int)
    return pd.DataFrame({"syllable_id": tokens, "segment": np.zeros(len(tokens), dtype=int),
                         "start_frame": starts, "end_frame_exclusive": starts + lengths,
                         "duration_s": lengths / 10., "extra": np.arange(len(tokens))})


def _data(*, shuffled=False):
    animals = ["a", "b", "c", "d"]
    sessions = pd.DataFrame({"session_id": [f"record_{a}" for a in animals], "animal_id": animals})
    tables = {}
    for i, animal in enumerate(animals):
        tokens = np.tile([0, 1, 2], 240)
        if shuffled:
            tokens = np.random.default_rng(100 + i).permutation(tokens)
        tables[f"record_{animal}"] = _bouts(tokens)
    folds = [["a", "b"], ["c", "d"]]
    return SimpleNamespace(sessions=sessions, sessions_of={a: [f"record_{a}"] for a in animals},
                           pool=animals, test=["reserved"], folds=folds, base_seed=19, vocab={"L": 3},
                           bouts=lambda *_: tables,
                           fold_split=lambda fold: ([a for a in animals if a not in folds[fold]], folds[fold]))


def _config(method="exact"):
    config = {"level": "L", "min_bout_ms": 0, "method": method, "chain_len": 3, "n_units": 1}
    config["min_prevalence" if method == "exact" else "penalty"] = 1 if method == "exact" else 0.
    return config


def test_filter_retains_bounds_and_breaks_removed_bout_and_native_gap():
    source = _bouts([0, 1, 0, 2], starts=[0, 10, 11, 30], lengths=[10, 1, 10, 10])
    original = source.copy(deep=True)
    filtered = filter_short_bouts(source, .5)
    assert filtered.syllable_id.tolist() == [0, 0, 2]
    assert filtered.segment.tolist() == [0, 1, 2]
    assert filtered.start_frame.tolist() == [0, 11, 30]
    assert filtered.end_frame_exclusive.tolist() == [10, 21, 40]
    assert filtered.duration_s.tolist() == [1., 1., 1.]
    assert filtered.extra.tolist() == [0, 2, 3]
    pd.testing.assert_frame_equal(source, original)
    unfiltered = filter_short_bouts(source, 0)
    assert unfiltered.segment.tolist() == [0, 0, 0, 1]
    assert motif_masks([sub.syllable_id.to_numpy() for _, sub in filtered.groupby("segment")], [(0, 0)]).sum() == 0


def test_filter_rejects_overlapping_native_intervals():
    with pytest.raises(ValueError, match="nonoverlapping"):
        filter_short_bouts(_bouts([0, 1], starts=[0, 1]), 0)


def test_vectorized_matcher_uses_greedy_nonoverlap_and_segment_boundaries():
    assert motif_hits([0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0]).tolist() == [0, 3]
    assert motif_hits([0, 0, 0, 0], [0, 0]).tolist() == [0, 2]
    assert motif_hits([0], [0, 1]).size == 0
    masks = motif_masks([np.array([0, 1]), np.array([2, 0, 1, 2])], [(0, 1, 2)])
    assert masks.tolist() == [[False, False, False, True, True, True]]


def test_dead_fraction_handles_majority_zero_and_all_zero_factors():
    assert activation_dead_fraction([10, 0, 0, 0]) == .75
    assert activation_dead_fraction([0, 0]) == 1.
    assert activation_dead_fraction([100, 100, .01]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        activation_dead_fraction([np.nan, 1])


def test_exact_known_routine_beats_shuffle_without_reading_reserved_recording():
    config = _config()
    real = evaluate_configuration(_data(), config, 0, per_animal=600, null_repeats=8)
    shuffled = evaluate_configuration(_data(shuffled=True), config, 0, per_animal=600, null_repeats=8)
    assert real["order_gain"] > .75
    assert abs(shuffled["order_gain"]) < .25
    assert real["order_gain"] > shuffled["order_gain"] + .5
    # Random chunk phase can censor up to two bouts at each boundary.
    assert real["explained"] >= 596 / 600
    assert real["heldout_animals"] == 4
    assert real["same_data_seed_stability"] is None
    assert real["sampled_heldout_bouts"] == 2400
    assert json.loads(json.dumps(real, allow_nan=False)) == real
    assert all("reserved" not in str(fold) for fold in real["details"]["support_by_fold"])
    repeated = evaluate_configuration(_data(), config, 0, per_animal=600, null_repeats=8)
    assert real == repeated


def test_seqnmf_uses_same_data_alternate_seed_and_deterministic_evaluation():
    config = _config("seqnmf")
    real = evaluate_configuration(_data(), config, 0, iterations=40, per_animal=90, null_repeats=3)
    shuffled = evaluate_configuration(_data(shuffled=True), config, 0, iterations=40, per_animal=90, null_repeats=3)
    assert real["order_gain"] > .2
    assert real["order_gain"] > shuffled["order_gain"] + .1
    assert 0 <= real["same_data_seed_stability"] <= 1.00001
    assert real["stability"] == pytest.approx((real["same_data_seed_stability"] + real["crossfold_stability"]) / 2)
    assert real["details"]["metric"] == "uncentered_reconstruction_score"
    for fold in real["details"]["support_by_fold"]:
        assert fold["seeds"]["model"] != fold["seeds"]["alternate"]
        assert all(len(row["null_nrmse"]) == 3 for row in real["details"]["animal_readouts_by_fold"][fold["fold"]])
    repeated = evaluate_configuration(_data(), config, 0, iterations=40, per_animal=90, null_repeats=3)
    assert real == repeated


def test_inactive_knobs_and_fold_leakage_are_rejected():
    data, config = _data(), _config()
    with pytest.raises(ValueError, match="only discovery"):
        evaluate_configuration(data, dict(config, history_lag=3), 0)
    data.test = ["a"]
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_configuration(data, config, 0)


def test_exact_coverage_weights_animals_equally_despite_different_exposure():
    data, config = _data(), _config()
    tables = {"record_a": _bouts(np.tile([0, 1, 2], 10)),
              "record_b": _bouts(np.zeros(720, dtype=int)),
              "record_c": _bouts(np.tile([0, 1, 2], 240)),
              "record_d": _bouts(np.tile([0, 1, 2], 240))}
    data.bouts = lambda *_: tables
    result = evaluate_configuration(data, config, 0, per_animal=600, null_repeats=3)
    rows = [row for fold in result["details"]["animal_readouts_by_fold"] for row in fold]
    animal_mean = np.mean([row["real_coverage"] for row in rows])
    exposure = {animal: evidence["sampled_bouts"] for fold in result["details"]["support_by_fold"]
                for animal, evidence in fold["heldout"].items()}
    bout_weighted = sum(row["real_coverage"] * exposure[row["animal_id"]] for row in rows) / sum(exposure.values())
    assert result["explained"] == pytest.approx(animal_mean)
    assert abs(animal_mean - bout_weighted) > .15


def test_short_supported_segments_are_sampled_and_missing_support_is_explicit():
    data, config = _data(), _config()
    tiny = {sid: _bouts([0, 1, 2, 0, 1, 2]) for sid in data.sessions.session_id}
    data.bouts = lambda *_: tiny
    result = evaluate_configuration(data, config, 0, per_animal=10, null_repeats=2)
    assert result["sampled_heldout_bouts"] == 24
    tiny["record_a"] = _bouts([0, 1])
    with pytest.raises(ValueError, match="no supported"):
        evaluate_configuration(data, config, 0, per_animal=10, null_repeats=2)
