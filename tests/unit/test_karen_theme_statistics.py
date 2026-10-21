"""Independent mouse-level, multiplicity and opportunity-weight expectations."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.theme_statistics import (
    _Assignments,
    _independent,
    _paired,
    infer,
)


def endpoint(waves=("3w_PI", "8w_PI")):
    rows = []
    for group, offset in [("Monomer", 0), ("PFF", 4)]:
        for animal in range(3):
            for wave_index, wave in enumerate(waves):
                rows.append({"session_id": f"{group}{animal}_{wave}", "animal_id": f"{group}{animal}",
                             "group": group, "wave": wave, "family": "recruitment", "theme": "theme_a",
                             "feature": "rate", "value": animal + offset + wave_index * 2,
                             "support": animal + 1, "opportunities": 60})
    return pd.DataFrame(rows)


def test_exact_partitions_and_welch_interval_have_independent_expectations():
    panel = endpoint().query("wave == '3w_PI'")
    result = _independent(panel, _Assignments(99, 4, 1024 ** 2))
    assert result["difference"] == 4
    assert result["p_raw"] == .1  # Two maximally separated partitions out of 6 choose 3.
    assert result["permutation_exact"] and result["permutation_draws"] == 20
    assert result["n_monomer"] == result["n_pff"] == 3
    assert result["ci_low"] < 4 < result["ci_high"]
    assert result["opportunities_monomer"] == 180


def test_exact_sign_flips_pair_animals_and_retain_constant_nonzero_changes():
    panel = pd.DataFrame({"animal_id": ["a", "b", "c"], "value": [2., 2., 2.]})
    result = _paired(panel, _Assignments(99, 4, 1024 ** 2))
    assert result["mean_change"] == result["ci_low"] == result["ci_high"] == 2
    assert result["p_raw"] == .25  # Both all-positive and all-negative out of 8 sign patterns.
    assert result["n_pairs"] == 3 and result["permutation_draws"] == 8


def test_ratio_mouse_wave_uses_opportunities_before_equal_wave_pooling(tmp_path):
    data = endpoint()
    data["aggregation"] = "ratio"
    data["numerator"] = data.value
    data["denominator"] = 1.
    extra = data.iloc[[0]].copy()
    extra["session_id"] = "extra_long_recording"
    extra["numerator"], extra["denominator"], extra["value"] = 100., 10., 10.
    data = pd.concat([data, extra], ignore_index=True)
    result = infer(data, tmp_path, replicates=9)
    panel = pd.read_csv(result["paths"]["mouse_readouts.csv"])
    actual = panel.query("animal_id == 'Monomer0' and wave == '3w_PI'").iloc[0]
    assert actual.value == pytest.approx(100 / 11)
    pooled = pd.read_csv(result["paths"]["pooled_mouse_readouts.csv"])
    actual = pooled.query("animal_id == 'Monomer0' and scope == 'pooled_untreated'").iloc[0]
    assert actual.value == pytest.approx((100 / 11 + 2) / 2)
    assert actual.n_waves == 2 and actual.n_recordings == 3
    complete = pooled.query("scope == 'pooled_untreated_complete'")
    assert complete.value.isna().all()
    contrasts = pd.read_csv(result["paths"]["contrasts.csv"])
    assert "all_waves_descriptive" not in set(contrasts.scope)


def test_paired_matching_uses_mouse_ids_not_row_order(tmp_path):
    data = endpoint().sample(frac=1, random_state=6)
    result = infer(data, tmp_path, replicates=9)
    changes = pd.read_csv(result["paths"]["paired_mouse_changes.csv"])
    assert len(changes) == 6 and np.all(changes.value == 2)
    longitudinal = pd.read_csv(result["paths"]["longitudinal.csv"])
    eight = longitudinal.query("target_wave == '8w_PI'")
    assert list(eight.mean_change) == [2, 2] and list(eight.p_raw) == [.25, .25]
    assert list(eight.opportunities_baseline) == [180, 180]
    assert list(eight.opportunities_followup) == [180, 180]
    interactions = pd.read_csv(result["paths"]["contrasts.csv"])
    eight = interactions.query("contrast_type == 'difference_in_changes' and scope == '3w_PI_to_8w_PI'").iloc[0]
    assert eight.difference == 0 and eight.p_raw == 1


def test_multiplicity_includes_all_endpoints_and_both_inference_types(tmp_path):
    data = endpoint()
    other = data.copy()
    other["feature"], other["theme"] = "fraction", "theme_b"
    other["value"] = 1.
    result = infer(pd.concat([data, other], ignore_index=True), tmp_path, replicates=9)
    tests = pd.read_csv(result["paths"]["all_inference.csv"])
    valid = tests.dropna(subset=["p_raw"])
    assert result["n_tests"] == len(valid)
    assert np.all(valid.global_test_count == len(valid))
    first = valid.query("scope == '3w_PI' and feature == 'rate'").iloc[0]
    assert first.p_raw == .1 and first.q_family == .2 and first.family_test_count == 2
    # Independent reference BH over every finite primary test, including paired changes.
    ordered = np.argsort(valid.p_raw.to_numpy())
    raw = valid.p_raw.to_numpy()[ordered]
    expected = np.minimum.accumulate((raw * len(raw) / np.arange(1, len(raw) + 1))[::-1])[::-1]
    np.testing.assert_allclose(valid.q_global.to_numpy()[ordered], np.minimum(expected, 1))
    assert set(valid.contrast_type) == {"group_difference", "paired_change", "difference_in_changes"}


def test_sparse_conditional_nan_does_not_become_zero_or_count_as_available_wave(tmp_path):
    data = endpoint(waves=("Ldopa",))
    data.loc[data.animal_id == "PFF0", "value"] = np.nan
    result = infer(data, tmp_path, replicates=9)
    contrasts = pd.read_csv(result["paths"]["contrasts.csv"])
    actual = contrasts.query("scope == 'Ldopa'").iloc[0]
    assert actual.n_pff == 2 and actual.status == "insufficient_animals" and pd.isna(actual.p_raw)
    assert actual.ldopa_age_treatment_confounded
    pooled = pd.read_csv(result["paths"]["pooled_mouse_readouts.csv"])
    assert pooled.query("scope == 'all_waves_descriptive' and animal_id == 'PFF0'").n_waves.iloc[0] == 0
    policy = json.loads((tmp_path / "inference_policy.json").read_text())
    assert policy["workers"] == 1 and policy["n_animals"] == 6


def test_assignment_cache_is_deterministic_and_reuses_fixed_mouse_partitions():
    animals = [str(i) for i in range(21)]
    first = _Assignments(299, 41, 1024 ** 2)
    weights, exact, _ = first.get(animals, 10)
    again, _, _ = first.get(animals, 10)
    other, _, _ = _Assignments(299, 41, 1024 ** 2).get(animals, 10)
    assert not exact and weights is again
    np.testing.assert_array_equal(weights, other)
    assert np.allclose(weights.sum(axis=1), 0)
    assert np.all(np.sum(weights > 0, axis=1) == 10)
    # These are mouse assignments, never an independent row shuffle over repeated waves.
    assert weights.shape == (299, 21)


@pytest.mark.parametrize("kind", ["group", "session", "duplicate", "ratio"])
def test_bad_identity_and_ratio_inputs_are_rejected_before_output(tmp_path, kind):
    data = endpoint()
    if kind == "group":
        data.loc[0, "group"] = "PFF"
    elif kind == "session":
        data.loc[0, "session_id"] = data.iloc[-1].session_id
    elif kind == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    else:
        data["aggregation"], data["numerator"], data["denominator"] = "ratio", 1., 0.
    with pytest.raises(ValueError):
        infer(data, tmp_path / "must_not_exist", replicates=9)
    assert not (tmp_path / "must_not_exist").exists()


def test_assignment_exact_enumeration_matches_all_labelings():
    animals = list("abcdef")
    weights, exact, total = _Assignments(2, 42, 1024 ** 2).get(animals, 3)
    expected = []
    values = np.arange(6.)
    for second in itertools.combinations(range(6), 3):
        first = sorted(set(range(6)) - set(second))
        expected.append(values[list(second)].mean() - values[first].mean())
    assert exact and total == 20
    np.testing.assert_allclose(weights @ values, expected, atol=1e-15)
