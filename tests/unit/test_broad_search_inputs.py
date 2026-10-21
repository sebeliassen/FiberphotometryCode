"""Native feature joins, frozen animal policy and matched quantitative windows."""

import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from studies.sequence_viz.broad_search_inputs import (
    _validate_reviewed_mask,
    generate_recipes,
    job_identity,
    join_verified_native_features,
    load_full_held_records,
    load_job_records,
    stratified_animal_folds,
)


def test_review_policy_accepts_only_unmodified_absent_mask_and_exact_exclusions():
    data = dict(frames=np.arange(6), labels=np.array([-1, -1, 2, 3, 4, 5]),
                analysis_excluded=np.array([True, True, False, False, False, False]))
    _validate_reviewed_mask(data, [(0, 2)])
    with pytest.raises(ValueError, match="differs"):
        _validate_reviewed_mask(data, [(0, 3)])
    del data["analysis_excluded"]
    _validate_reviewed_mask(data, [])
    with pytest.raises(ValueError, match="lacks"):
        _validate_reviewed_mask(data, [(0, 2)])


def _join_inputs():
    frames = np.array([10, 11, 12, 20, 21, 22])
    native = dict(native_frames=frames, labels=np.array([4, 4, 5, 6, 6, 7]),
                  centroid_x=frames.astype(float), centroid_y=np.zeros(6))
    sparse = dict(table_index=np.array([10, 12, 20, 22]), labels=np.array([4, 5, 6, 7]))
    sparse.update({f"latent_state_{i}": np.arange(4, dtype=float)+100*i for i in range(10)})
    dense = dict(table_index=frames.copy(), frame_index=frames.copy(), labels=native["labels"].copy(),
                 centroid_x=native["centroid_x"].copy(), centroid_y=native["centroid_y"].copy(),
                 heading=np.linspace(-1., 1., 6))
    return native, sparse, dense


def test_native_latents_join_only_supported_partner_and_dense_heading_is_exact():
    native, sparse, dense = _join_inputs()
    result = join_verified_native_features(native, sparse, dense)
    np.testing.assert_array_equal(result["frames"], [10, 11, 12, 20, 21, 22])
    np.testing.assert_array_equal(result["latent"][:, 0], [0., 0., 1., 2., 2., 3.])
    np.testing.assert_array_equal(result["heading"], dense["heading"])
    assert result["join_counts"] == {"exact_native_rows": 4, "supported_partner_rows": 2}
    native["native_frames"][2] = 13
    with pytest.raises(ValueError, match="unsupported sparse gap"):
        join_verified_native_features(native, sparse, dense)


def test_native_join_rejects_wrong_labels_or_inferred_dense_coordinates():
    native, sparse, dense = _join_inputs()
    sparse["labels"][0] = 99
    with pytest.raises(ValueError, match="sparse model"):
        join_verified_native_features(native, sparse, dense)
    native, sparse, dense = _join_inputs()
    dense["frame_index"] = dense["frame_index"]*2
    with pytest.raises(ValueError, match="disagree"):
        join_verified_native_features(native, sparse, dense)


def test_animal_strata_have_no_leakage_and_balance_small_groups():
    groups = {str(i): "KI" if i < 7 else "WT" for i in range(15)}
    folds = stratified_animal_folds(groups)
    assert folds == stratified_animal_folds(dict(reversed(list(groups.items()))))
    assert sorted(map(len, folds)) == [5, 5, 5]
    assert len({subject for fold in folds for subject in fold}) == 15
    assert sorted(sum(groups[s] == "KI" for s in fold) for fold in folds) == [2, 2, 3]


def _window_manifest(tmp_path):
    records = []
    for subject in ("a", "b", "c", "reserved"):
        for recording in (subject+"_one", subject+"_two"):
            path = tmp_path/(recording+".npz")
            # Identical labels either side of the gap remain separate source bouts.
            starts = np.r_[np.arange(0, 20)*2, 100+np.arange(12)*2]
            stops = starts+2
            np.savez_compressed(path, start_frames=starts, stop_frames=stops,
                parent_ids=np.tile(["0", "1"], 16), values=np.arange(32)[:, None].astype(float),
                feature_names=np.array(["duration_s"]))
            stat = path.stat()
            records.append(dict(subject=subject, recording=recording, npz_path=str(path), sha256="frozen_"+recording,
                                size=stat.st_size, mtime_ns=stat.st_mtime_ns, bout_count=32,
                                metadata={"genotype": "not_for_model", "drug": "not_for_model"},
                                exclude_from_search=subject == "reserved"))
    return dict(schema_version=1, manifest_hash="frozen", cohorts=[dict(cohort_id="synthetic", model_id="source",
        clock="native", fps=60., vocabulary=["0", "1"], annotation_mapping={"0": "A", "1": "B"},
        minimum_support=.8, exploration_subjects=["a", "b", "c"], reserved_subjects=["reserved"],
        folds=[["a"], ["b"], ["c"]], records=records)])


def test_shared_windows_use_original_intervals_and_cap_each_animal(tmp_path):
    manifest = _window_manifest(tmp_path)
    records = load_job_records(manifest, "synthetic", 0, per_animal=30, held_per_animal=20, window_seed=73)
    assert Counter(record["split"] for record in records) == {"held": 2, "train": 4}
    assert {record["subject"] for record in records} == {"a", "b", "c"}
    events = Counter()
    for record in records:
        events[record["subject"]] += record["achieved_events"]
        assert not {"genotype", "group", "drug", "metadata", "DA"} & set(record)
        assert np.all(record["start_frames"][1:] == record["stop_frames"][:-1])
    assert events == {"a": 20, "b": 30, "c": 30}
    manifest["cohorts"][0]["records"].reverse()
    again = load_job_records(manifest, "synthetic", 0, per_animal=30, held_per_animal=20, window_seed=73)
    for first, second in zip(records, again):
        np.testing.assert_array_equal(first["source_indices"], second["source_indices"])
    identity = job_identity(manifest, "synthetic", 0, records, window_seed=73, per_animal=30, held_per_animal=20)
    assert len(identity["windows"]) == 6 and identity["held_subjects"] == ["a"]
    json.dumps(identity, allow_nan=False)


def test_frozen_inputs_reject_changes_and_reserved_fold_leakage(tmp_path):
    manifest = _window_manifest(tmp_path)
    manifest["cohorts"][0]["records"][0]["size"] += 1
    with pytest.raises(ValueError, match="changed"):
        load_job_records(manifest, "synthetic", 0, per_animal=30, held_per_animal=20, window_seed=73)


def test_full_held_loader_retains_all_bouts_gaps_and_original_animal_fold(tmp_path):
    manifest = _window_manifest(tmp_path)
    selected = load_full_held_records(manifest, "synthetic", 0)
    assert len(selected) == 2 and {r["subject"] for r in selected} == {"a"}
    for r in selected:
        assert r["split"] == "held" and len(r["start_frames"]) == 32
        np.testing.assert_array_equal(r["source_indices"], np.arange(32))
        assert r["start_frames"][20] == 100 and r["stop_frames"][19] == 40
        assert not {"metadata", "genotype", "drug", "DA"} & set(r)
    manifest["cohorts"][0]["folds"][0].append("reserved")
    with pytest.raises(ValueError, match="reserved"):
        load_full_held_records(manifest, "synthetic", 0)
    manifest = _window_manifest(tmp_path)
    manifest["cohorts"][0]["folds"][0].append("reserved")
    with pytest.raises(ValueError, match="frozen animal folds"):
        load_job_records(manifest, "synthetic", 0, per_animal=30, held_per_animal=20, window_seed=73)


def test_conditional_search_is_balanced_and_never_reuses_annotation_for_new_vocab():
    recipes = generate_recipes()
    assert len(recipes) == 120 and recipes == generate_recipes()
    assert set(Counter((r["representation_cell"], r["method"]) for r in recipes).values()) == {4}
    assert len({r["base_id"] for r in recipes}) == 120
    for recipe in recipes:
        if recipe["route"] not in {"kpms", "parallel"}:
            assert recipe["resolution"] != "annotation"
        if recipe["route"] == "parallel":
            assert recipe["bundle"] in {"movement", "posture", "movement_posture"}
        if recipe["method"] != "hsmm":
            assert recipe["initialization"] == "configured"
            assert "learn_durations" not in recipe["active_parameters"]
        else:
            assert recipe["initialization"] in {"configured", "dirichlet"}
        if recipe["method"] in {"exact", "tolerant", "grammar"}:
            assert recipe["feature_weight"] == 1.
            assert "feature_weight" not in recipe["active_parameters"]
        if recipe["resolution"] == "quantitative":
            assert "merge_clusters" in recipe["active_parameters"]
            if recipe["route"] in {"feature_only", "joint"}:
                assert recipe["merge_clusters"] < recipe["clusters"]
        else:
            assert "merge_clusters" not in recipe["active_parameters"]
        assert "genotype" not in recipe and "DA" not in recipe
    assert any(r["bundle"] == "spatial" and r["route"] != "parallel" for r in recipes)
    with pytest.raises(ValueError, match="multiple of thirty"):
        generate_recipes(recipes_per_cohort=31)


def test_import_and_recipe_generation_do_not_load_numeric_or_model_backends():
    code = ("import sys; from studies.sequence_viz.broad_search_inputs import generate_recipes; "
            "generate_recipes(); assert not any(n in sys.modules for n in "
            "('numpy','pandas','jax','torch','keypoint_moseq'))")
    # The shared lab interpreter's site setup already imports NumPy; -S isolates
    # this module's import contract from those environment startup side effects.
    completed = subprocess.run([sys.executable, "-B", "-S", "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
