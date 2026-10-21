"""Integrated recipe invariants with independent native bounds and controls."""

import json
from copy import deepcopy

import numpy as np
import pytest

from fpcode.workflows.sequence_search import compare_recipe_catalogues, evaluate_recipe


def records():
    result = []
    for animal, split in (("a", "train"), ("b", "train"), ("c", "held"), ("d", "held")):
        labels = np.tile(["0", "1", "2", "3"], 15)
        starts = np.arange(len(labels), dtype=np.int64)*10
        starts[30:] += 100
        values = np.column_stack([np.ones(len(labels)), labels.astype(float)+.1,
                                  np.arange(len(labels)) % 4, np.arange(len(labels)) % 7])
        result.append(dict(cohort="fixture", recording="rec_"+animal, subject=animal,
            split=split, clock="native", fps=10., source_vocabulary=["0", "1", "2", "3"],
            start_frames=starts, stop_frames=starts+10, parent_ids=labels,
            feature_names=("duration_s", "mean_speed_px_s", "centroid_x_px", "centroid_y_px"), values=values))
    return result


def recipe(method="exact", route="kpms", role="describe"):
    return dict(method=method, route=route, feature_role=role, bundle="movement", resolution="identity",
        length=3, factors=2, min_prevalence=1, seed=7, feature_weight=1., clusters=3,
        creator_seed=19, parent_weight=1., iterations=3, penalty=.01, activation_quantile=.7,
        max_error=2., phase_count=3, phase_max=3, background_max=12, entry_probability=.1,
        early_exit_probability=0., skip_probability=0., reentry=True, extra_background=True,
        learn_durations=True, initialization="configured", pseudocount=.1, variance_floor=.1,
        hsmm_targets=1, edit_budget=1, gate_multiplier=2., grammar_optional=True, grammar_repeat=2)


def identity():
    return dict(source_model_id="fixture_model", cohort="fixture", minimum_support=.8,
                vocabulary=["0", "1", "2", "3"], annotation_mapping={"0": "move", "1": "move", "2": "turn", "3": "rest"},
                verified_input_fingerprint="synthetic_v1", fold=0)


@pytest.mark.parametrize("method", ["exact", "tolerant", "seqnmf", "hsmm", "grammar"])
@pytest.mark.parametrize("route,role", [("kpms", "describe"), ("kpms", "episode"), ("feature_only", "describe"),
                                       ("within_kpms", "describe"), ("joint", "describe"), ("parallel", "describe")])
def test_all_routes_and_detectors_preserve_support(method, route, role, tmp_path):
    inputs = records()
    before = deepcopy(inputs)
    result = evaluate_recipe(inputs, recipe(method, route, role), identity(), tmp_path, null_repeats=1)
    assert result["status"] in {"SUCCESS", "EMPTY"}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["summary"]["held_animals"] == 2
    assert 0 <= result["summary"]["coverage_fraction"] <= 1
    assert {c["kind"] for c in result["controls"]} == {"shuffle", "first_order"}
    for row in result["records"]:
        for call in row["raw_calls"]:
            assert call["start_frame"] % 10 == call["stop_frame"] % 10 == 0
            assert not (call["start_frame"] < 300 and call["stop_frame"] > 300)
            assert not (call["start_frame"] < 400 and call["stop_frame"] > 400)
        assert row["covered_exposure_s"] <= row["available_exposure_s"]
    for r, b in zip(inputs, before):
        np.testing.assert_array_equal(r["values"], b["values"])
        np.testing.assert_array_equal(r["start_frames"], b["start_frames"])
    if role == "episode" or route == "parallel":
        assert result["conditions"]["consumed_features"]


def test_descriptive_features_reuse_fit_and_categorical_calls(tmp_path):
    first = evaluate_recipe(records(), recipe(), identity(), tmp_path, null_repeats=0)
    changed = recipe()
    changed["bundle"] = "spatial"
    second = evaluate_recipe(records(), changed, identity(), tmp_path, null_repeats=0)
    assert second["cache"]["representation_hit"] and second["cache"]["model_hit"]
    assert first["cache"]["model_key"] == second["cache"]["model_key"]
    assert first["records"][0]["raw_calls"] == second["records"][0]["raw_calls"]
    assert not second["quality"]["physical"]["features_used_in_construction"]


def test_empty_hsmm_proposal_is_a_valid_empty_result(tmp_path):
    configured = recipe("hsmm")
    configured["min_prevalence"] = 3
    result = evaluate_recipe(records(), configured, identity(), tmp_path, null_repeats=0)
    assert result["status"] == "EMPTY"
    assert result["summary"]["selected_calls"] == 0
    assert "prevalence" in result["conditions"]["empty_reason"]


def test_missing_direct_features_leave_barriers_and_full_exposure(tmp_path):
    inputs = records()
    for r in inputs:
        r["values"][3, 1] = np.nan
    result = evaluate_recipe(inputs, recipe(role="episode"), identity(), tmp_path, null_repeats=0)
    for r in result["records"]:
        assert r["removed_feature_bouts"] == 1
        assert r["available_exposure_s"] == 59 and r["source_exposure_s"] == 60
        assert all(not (c["start_frame"] < 40 and c["stop_frame"] > 30) for c in r["raw_calls"])


def test_animal_leakage_is_rejected_before_fit(tmp_path):
    inputs = records()
    inputs[-1]["subject"] = "a"
    with pytest.raises(ValueError, match="leakage"):
        evaluate_recipe(inputs, recipe(), identity(), tmp_path, null_repeats=0)
    assert not list(tmp_path.iterdir())


def test_parallel_execution_bundle_cannot_duplicate_spatial_view(tmp_path):
    cfg = recipe(route="parallel")
    cfg["bundle"] = "spatial"
    with pytest.raises(ValueError, match="nonspatial"):
        evaluate_recipe(records(), cfg, identity(), tmp_path, null_repeats=0)


def test_seed_stability_matches_native_overlap_instead_of_numerical_ids():
    common = dict(recording="rec", subject="mouse", clock="native", fps=10., valid_intervals=[[0, 40]])
    first = dict(records=[dict(common, selected_calls=[dict(type_id="factor_0", start_frame=0, stop_frame=10),
                                                      dict(type_id="factor_1", start_frame=20, stop_frame=30)])])
    second = dict(records=[dict(common, selected_calls=[dict(type_id="factor_1", start_frame=0, stop_frame=10),
                                                       dict(type_id="factor_0", start_frame=20, stop_frame=30)])])
    compared = compare_recipe_catalogues(first, second)
    assert compared["matched_type_temporal_f1"] == 1.
    assert compared["mapping"][0]["first_type"] != compared["mapping"][0]["second_type"]
    second["records"][0]["selected_calls"][1]["stop_frame"] = 40
    assert compare_recipe_catalogues(first, second)["matched_type_temporal_f1"] == pytest.approx(.8)


def test_seed_stability_empty_is_missing_and_support_mismatch_rejected():
    r = dict(records=[dict(recording="r", subject="m", clock="native", fps=10., valid_intervals=[[0, 10]], selected_calls=[])])
    assert compare_recipe_catalogues(r, r)["matched_type_temporal_f1"] is None
    second = deepcopy(r)
    second["records"][0]["clock"] = "other"
    with pytest.raises(ValueError, match="clock"):
        compare_recipe_catalogues(r, second)


def test_quantitative_resolution_can_coarsen_a_feature_codebook(tmp_path):
    cfg = recipe(route="feature_only")
    cfg.update(resolution="quantitative", merge_clusters=1)
    result = evaluate_recipe(records(), cfg, identity(), tmp_path, null_repeats=0)
    assert len(result["representation"]["catalog"]) == 1
    assert result["representation"]["configuration"]["merge_clusters"] == 1
    assert result["summary"]["feature_support_fraction"] == 1


def test_empty_complete_training_features_make_first_order_control_unavailable(tmp_path):
    inputs = records()
    for r in inputs:
        if r["split"] == "train":
            r["values"][::2, 0] = np.nan
            r["values"][1::2, 1] = np.nan
    result = evaluate_recipe(inputs, recipe(role="episode"), identity(), tmp_path, null_repeats=1)
    assert result["status"] == "EMPTY"
    null = next(c for c in result["controls"] if c["kind"] == "first_order")
    assert null["status"] == "UNAVAILABLE"


def test_inert_gate_weight_and_unknown_search_axis_are_rejected(tmp_path):
    cfg = recipe(role="episode")
    cfg["feature_weight"] = 2.
    with pytest.raises(ValueError, match="no independent feature-weight"):
        evaluate_recipe(records(), cfg, identity(), tmp_path, null_repeats=0)
    cfg = dict(recipe(), magic_metric=True)
    with pytest.raises(ValueError, match="unsupported recipe"):
        evaluate_recipe(records(), cfg, identity(), tmp_path, null_repeats=0)


@pytest.mark.parametrize("method,route,role", [
    ("exact", "kpms", "describe"), ("tolerant", "kpms", "episode"),
    ("grammar", "kpms", "describe"), ("seqnmf", "parallel", "describe"),
    ("seqnmf", "feature_only", "describe"), ("hsmm", "within_kpms", "describe"),
    ("hsmm", "kpms", "episode"), ("exact", "joint", "describe")])
def test_frozen_inference_reproduces_calls_without_fit_or_cache_write(method, route, role, tmp_path, monkeypatch):
    from fpcode.behavior.primitives.segmentation import WholeBoutCodebook
    from fpcode.workflows.sequence_search import predict_frozen_recipe

    cfg, inputs = recipe(method, route, role), records()
    result = evaluate_recipe(inputs, cfg, identity(), tmp_path, null_repeats=0)
    with np.load(tmp_path/(result["cache"]["model_key"]+".npz"), allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files if k != "__identity_json"}
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.npz")}
    def forbid(*_args, **_kwargs):
        raise AssertionError("inference must not fit")
    monkeypatch.setattr(WholeBoutCodebook, "fit", forbid)
    for r, expected in zip([r for r in inputs if r["split"] == "held"], result["records"]):
        predicted = predict_frozen_recipe(r, cfg, result["representation"], arrays, result["conditions"],
            representation_id=result["cache"]["representation_key"], model_id=result["cache"]["model_key"])
        assert predicted["raw_calls"] == expected["raw_calls"]
        assert predicted["selected_calls"] == expected["selected_calls"]
        assert predicted["valid_intervals"] == expected["valid_intervals"]
        if method == "hsmm":
            selected = {(c["type_id"], c["start_frame"], c["stop_frame"]) for c in predicted["selected_calls"]}
            for stage in predicted["diagnostics"].get("stages", []):
                assert stage["phases"]
                assert all(stage["start_frame"] <= p["start_frame"] < p["stop_frame"] <= stage["stop_frame"]
                           for p in stage["phases"])
                assert stage["completion"]["evidence_kind"] == "selected_model_path"
            assert selected <= {(s["type_id"], s["start_frame"], s["stop_frame"])
                                for s in predicted["diagnostics"].get("stages", [])}
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.npz")}


def test_frozen_inference_rejects_mismatched_conditions_and_honours_array_cap(tmp_path):
    from fpcode.workflows.sequence_search import predict_frozen_recipe

    cfg = recipe("seqnmf")
    result = evaluate_recipe(records(), cfg, identity(), tmp_path, null_repeats=0)
    with np.load(tmp_path/(result["cache"]["model_key"]+".npz"), allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files if k != "__identity_json"}
    kw = dict(representation_id=result["cache"]["representation_key"], model_id=result["cache"]["model_key"])
    with pytest.raises(ValueError, match="method/catalogue"):
        predict_frozen_recipe(records()[-1], cfg, result["representation"], arrays,
                              dict(result["conditions"], method="exact"), **kw)
    with pytest.raises(MemoryError, match="array budget"):
        predict_frozen_recipe(records()[-1], cfg, result["representation"], arrays, result["conditions"],
                              max_working_bytes=1, **kw)
