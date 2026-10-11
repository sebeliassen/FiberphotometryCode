"""Small functional tests: real parsing/IO/orchestration, no model fitting or video encoding."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from fpcode.cli import evaluate, fuse, render, train
from fpcode.common.config import compute_config_hash
from fpcode.kpms.config import resolve_training_config
from fpcode.kpms.evaluation import evaluate_checkpoints
from fpcode.viz.grid_movies import _generate_selected_grid_movies, exact_grid_layout, generate_grid_movies
from fpcode.workflows.fusion import fusion_options
from fpcode.workflows.training import fit_model


@pytest.fixture(autouse=True)
def quiet_provenance(monkeypatch):
    import fpcode.common.run_record as run_record
    monkeypatch.setattr(run_record, "_get_gpu_info", lambda: {"available": False})
    monkeypatch.setattr(run_record, "_get_git_info", lambda: {"git_sha": "test", "git_dirty": False})
    for module in (evaluate, fuse, render, train):
        monkeypatch.setattr(module, "send_ntfy", lambda **kwargs: pytest.fail("Unexpected notification"))


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


def checkpoint(path, bounds=((0, 4),)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("model_snapshots/5/states/x", data=np.zeros((1, 4, 1)))
        handle.create_dataset("model_snapshots/5/states/z", data=np.array([[0, 1, 1]]))
        handle.create_dataset("model_snapshots/5/params/pi", data=np.eye(2))
        handle.create_dataset("metadata/arr0", data=np.array([b"recording"]))
        handle.create_dataset("metadata/arr1", data=np.asarray(bounds))
    return path


def results(path):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("recording/syllable", data=np.array([1, 1, 1, 0, 0, 0]))
        handle.create_dataset("recording/coordinates", data=np.arange(12).reshape(6, 1, 2))
        handle.create_dataset("recording/centroid", data=np.arange(12).reshape(6, 2))
        handle.create_dataset("recording/heading", data=np.zeros(6))
    return path


def test_nested_training_parameters_reach_fitting_boundary(tmp_path, monkeypatch):
    data = tmp_path / "preprocessed_data.pkl"
    data.write_bytes(b"fixture")
    config = write_yaml(tmp_path / "config.yaml", {
        "project_dir": str(tmp_path), "model_name": "tiny", "data_path": str(data), "seed": 8,
        "hyperparameters": {"num_ar_iters": 2, "num_slds_iters": 3, "total_iters": 5,
                            "save_every_n_iters": 1, "init_kappa": 17, "slds_kappa": 19},
    })
    calls = []

    def backend(cfg, model_dir, seed, device):
        calls.append((cfg, seed, device))
        model_dir.mkdir()
        output = model_dir / "checkpoint.h5"
        output.write_bytes(b"checkpoint")
        return output

    monkeypatch.setattr(train, "fit_model", lambda cfg, **kwargs: fit_model(cfg, backend=backend, **kwargs))
    args = train.parse_args(["--config", str(config), "--seed", "11", "--device", "cpu", "--no-notify"])
    assert train.run_training(args) == 0
    consumed, seed, device = calls[0]
    assert (consumed["num_ar_iters"], consumed["num_slds_iters"], consumed["total_iters"]) == (2, 3, 5)
    assert (consumed["save_every_n_iters"], consumed["init_kappa"], consumed["slds_kappa"]) == (1, 17, 19)
    assert (seed, device) == (11, "cpu")
    record = json.loads((tmp_path / "tiny/run.json").read_text())
    assert record["status"] == "SUCCESS"
    assert record["extra"]["resolved_config"]["num_slds_iters"] == 3
    assert record["config"]["seed"] == 11 and record["config"]["device"] == "cpu"
    assert record["config_hash"] == compute_config_hash(record["config"])
    assert record["source_config"] == {"path": str(config), "hash": compute_config_hash(yaml.safe_load(config.read_text()))}
    assert "config_path" not in record["config"] and "config_hash" not in record["config"]


def test_production_nested_settings_resolve_to_350_iterations(repo_root):
    cfg = yaml.safe_load((repo_root / "configs/training/cohort_static_21kp_v2_seed101.yaml").read_text())
    resolved = resolve_training_config(cfg)
    assert (resolved["num_ar_iters"], resolved["num_slds_iters"], resolved["total_iters"]) == (50, 300, 350)
    assert resolved["init_kappa"] == 7.5e13
    assert resolved["slds_kappa"] == 7.5e12


@pytest.mark.parametrize("config, message", [
    ({"num_ar_iters": 1, "hyperparameters": {"num_ar_iters": 2}}, "Conflicting"),
    ({"hyperparameters": {"num_slds_itrs": 3}}, "Unknown"),
    ({"num_ar_iters": -1}, "num_ar_iters"),
    ({"num_ar_iters": 1.5}, "num_ar_iters"),
    ({"num_ar_iters": True}, "num_ar_iters"),
    ({"total_iters": 251}, "total_iters"),
    ({"save_every_n_iters": 0}, "save_every"),
    ({"use_memory_patch": "false"}, "boolean"),
    ({"slds_kappa": float("nan")}, "finite"),
])
def test_invalid_training_settings_fail_explicitly(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_training_config(config)


def test_training_dry_run_creates_no_output(tmp_path, monkeypatch):
    data = tmp_path / "data.pkl"
    data.touch()
    cfg = write_yaml(tmp_path / "cfg.yaml", {"project_dir": str(tmp_path / "new"), "data_path": str(data)})
    monkeypatch.setattr(train, "fit_model", lambda *a, **k: pytest.fail("Dry run invoked fitting"))
    assert train.main(["--config", str(cfg), "--dry-run", "--no-notify"]) == 0
    assert not (tmp_path / "new").exists()


def test_training_missing_checkpoint_records_failure(tmp_path, monkeypatch):
    cfg = write_yaml(tmp_path / "cfg.yaml", {"project_dir": str(tmp_path), "model_name": "tiny"})
    monkeypatch.setattr(train, "fit_model", lambda *a, **k: tmp_path / "missing.h5")
    assert train.main(["--config", str(cfg), "--no-notify"]) == 1
    record = json.loads((tmp_path / "tiny/run.json").read_text())
    assert record["status"] == "FAILED"
    assert "missing checkpoint" in record["error_message"]


@pytest.mark.parametrize("count", [1, 2])
def test_actual_evaluate_parser_runs_tiny_checkpoints(tmp_path, count):
    checkpoints = [checkpoint(tmp_path / f"model{index}/checkpoint.h5") for index in range(count)]
    before = [path.read_bytes() for path in checkpoints]
    output = tmp_path / "evaluation"
    args = evaluate.parse_args(["--models", *map(str, checkpoints), "--output-dir", str(output), "--no-notify"])
    assert evaluate.run_evaluation(args) == 0
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "SUCCESS"
    assert record["extra"]["n_models"] == count
    assert record["config"]["checkpoints"] == list(map(str, checkpoints))
    assert record["config_hash"] == compute_config_hash(record["config"])
    for path in checkpoints:
        result = record["extra"]["results"][str(path)]
        assert result["reindex_order"] == [1, 0]
        assert result["max_frequency"] == pytest.approx(2 / 3)
    assert [path.read_bytes() for path in checkpoints] == before


def test_evaluate_default_output_dir_and_directory_input(tmp_path):
    path = checkpoint(tmp_path / "model/checkpoint.h5")
    assert evaluate.main(["--models", str(path.parent), "--no-notify"]) == 0
    assert json.loads((path.parent / "run.json").read_text())["status"] == "SUCCESS"


def test_evaluate_incompatible_dataset_records_failure(tmp_path):
    first = checkpoint(tmp_path / "first.h5")
    second = checkpoint(tmp_path / "second.h5", bounds=((0, 3),))
    output = tmp_path / "evaluation"
    assert evaluate.main(["--models", str(first), str(second), "--output-dir", str(output), "--no-notify"]) == 1
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "FAILED"
    assert "not comparable" in record["error_message"]


def test_evaluate_dry_run_writes_no_output(tmp_path):
    path = checkpoint(tmp_path / "checkpoint.h5")
    output = tmp_path / "not_created"
    assert evaluate.main(["--models", str(path), "--output-dir", str(output), "--dry-run", "--no-notify"]) == 0
    assert not output.exists()


def test_evaluation_excludes_padding_and_aligns_state_frames_after_lags(tmp_path):
    path = tmp_path / "padded.h5"
    with h5py.File(path, "w") as handle:
        handle["model_snapshots/5/states/x"] = np.zeros((2, 8, 1))
        # Two valid 1s in the first row, three valid 0s in the second. Everything
        # after each segment's actual length is padding, regardless of its label.
        handle["model_snapshots/5/states/z"] = np.array([[1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
        handle["model_snapshots/5/params/pi"] = np.eye(2)
        handle["metadata/arr0"] = np.array([b"short", b"longer"])
        handle["metadata/arr1"] = np.array([[0, 3], [10, 14]])
    before = path.read_bytes()
    actual = evaluate_checkpoints([path])[str(path)]
    assert actual["frequency_method"] == "masked_frame_occupancy"
    assert actual["evaluated_frames"] == 5
    assert actual["reindex_order"] == [0, 1]
    assert actual["max_frequency"] == pytest.approx(3 / 5)
    assert actual["active_states"] == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("bounds, message", [(((0, 1),), "no valid syllable"), (((0, 5),), "segment bounds")])
def test_evaluation_rejects_invalid_or_empty_state_masks(tmp_path, bounds, message):
    path = checkpoint(tmp_path / "checkpoint.h5", bounds=bounds)
    with pytest.raises(ValueError, match=message):
        evaluate_checkpoints([path])


def test_evaluation_ignores_invalid_padding_labels_but_rejects_invalid_active_labels(tmp_path):
    path = checkpoint(tmp_path / "checkpoint.h5", bounds=((0, 2),))
    with h5py.File(path, "r+") as handle:
        handle["model_snapshots/5/states/z"][...] = [[1, -1, 9]]
    actual = evaluate_checkpoints([path])[str(path)]
    assert actual["evaluated_frames"] == 1
    assert actual["reindex_order"] == [1, 0]
    assert actual["max_frequency"] == 1
    with h5py.File(path, "r+") as handle:
        handle["model_snapshots/5/states/z"][0, 0] = 9
    with pytest.raises(ValueError, match="state vocabulary"):
        evaluate_checkpoints([path])


def test_render_trajectories_writes_actual_selected_png(tmp_path):
    source = results(tmp_path / "results.h5")
    output = tmp_path / "figures"
    before = source.read_bytes()
    assert render.main(["--results", str(source), "--type", "trajectories", "--syllables", "1",
                        "--output-dir", str(output), "--no-notify"]) == 0
    png = output / "trajectories/syllable1.png"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not (output / "trajectories/syllable0.png").exists()
    assert not (output / "syllable_usage.png").exists()
    record = json.loads((output / "run.json").read_text())
    assert record["extra"]["output"]["files"] == [str(png)]
    assert record["config"]["syllables"] == [1]
    assert record["config_hash"] == compute_config_hash(record["config"])
    assert source.read_bytes() == before


def test_grid_request_without_video_manifest_fails_instead_of_usage_plot(tmp_path):
    source = results(tmp_path / "results.h5")
    output = tmp_path / "render"
    assert render.main(["--results", str(source), "--type", "grid_movies", "--output-dir", str(output), "--no-notify"]) == 1
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "FAILED"
    assert "video_paths" in record["error_message"]
    assert not (output / "syllable_usage.png").exists()


@pytest.mark.parametrize("tiles, shape", [(4, (2, 2)), (9, (3, 3)), (16, (4, 4)), (12, (3, 4)), (24, (4, 6))])
def test_grid_layout_contains_exact_requested_tiles(tiles, shape):
    assert exact_grid_layout(tiles) == shape
    assert np.prod(shape) == tiles


def test_render_dry_run_checks_requirements_and_creates_no_output(tmp_path):
    source = results(tmp_path / "results.h5")
    output = tmp_path / "not_created"
    assert render.main(["--results", str(source), "--type", "trajectories", "--output-dir", str(output),
                        "--dry-run", "--no-notify"]) == 0
    assert not output.exists()
    assert render.main(["--results", str(source), "--type", "grid_movies", "--output-dir", str(output),
                        "--dry-run", "--no-notify"]) == 1
    assert not output.exists()


def density_render_fixture(tmp_path, bouts, *, background_bouts=None, keypoints=1, sampling_mode="density"):
    if background_bouts is None:
        background_bouts = bouts
    assert background_bouts in (bouts, bouts - 1)
    middle = np.tile(np.repeat([0, 1], 3), background_bouts)
    if background_bouts < bouts:
        middle = np.r_[middle, np.zeros(3, dtype=int)]
    # Padding label 2 lies outside the displayed context. Both modeled syllables
    # have the explicitly requested number of eligible three-frame bouts.
    labels = np.r_[np.full(10, 2), middle, np.full(20, 2)]
    source = tmp_path / "density_results.h5"
    with h5py.File(source, "w") as handle:
        handle["recording/syllable"] = labels
        handle["recording/coordinates"] = np.zeros((len(labels), keypoints, 2))
        handle["recording/centroid"] = np.zeros((len(labels), 2))
        handle["recording/heading"] = np.zeros(len(labels))
    video = tmp_path / "placeholder.avi"
    video.write_bytes(b"readiness fixture; never decoded")
    manifest = write_yaml(tmp_path / "render.yaml", {"fps": 10, "sampling_mode": sampling_mode,
                          "video_paths": {"recording": str(video)}})
    output = tmp_path / "not_created"
    return ["--results", str(source), "--manifest", str(manifest), "--type", "grid_movies",
            "--syllables", "0", "--output-dir", str(output), "--dry-run", "--no-notify"], output


@pytest.mark.parametrize("mode, bouts, expected", [("density", 49, 1), ("density", 50, 0), ("random", 49, 0)])
def test_render_dry_run_enforces_density_neighbor_minimum_without_changing_mode(tmp_path, monkeypatch, caplog,
                                                                               mode, bouts, expected):
    args, output = density_render_fixture(tmp_path, bouts, sampling_mode=mode)
    monkeypatch.setattr(render, "render_model_visualizations", lambda *a, **k: pytest.fail("Dry run called backend"))
    assert render.main(args) == expected
    assert not output.exists()
    if expected:
        assert "at least 50 eligible bouts" in caplog.text


def test_render_density_dry_run_checks_unrequested_reference_syllables(tmp_path, caplog):
    args, output = density_render_fixture(tmp_path, 50, background_bouts=49)
    assert render.main(args) == 1
    assert "insufficient counts: {1: 49}" in caplog.text
    assert not output.exists()


def test_render_density_dry_run_rejects_grid_larger_than_neighbor_population(tmp_path, caplog):
    args, output = density_render_fixture(tmp_path, 64)
    assert render.main([*args, "--grid-size", "64"]) == 1
    assert "at most 50 grid tiles" in caplog.text
    assert not output.exists()


def test_render_density_dry_run_rejects_pca_with_no_pose_features(tmp_path, caplog):
    args, output = density_render_fixture(tmp_path, 50, keypoints=0)
    assert render.main(args) == 1
    assert "Density PCA requires" in caplog.text
    assert not output.exists()


def test_grid_backend_cannot_claim_old_movie_as_new_output(tmp_path, monkeypatch):
    import sys
    old_movie = tmp_path / "syllable1.mp4"
    old_movie.write_bytes(b"previous run")
    backend = SimpleNamespace(generate_grid_movies=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "keypoint_moseq", backend)
    output = generate_grid_movies({"recording": {"syllable": np.array([1, 1, 1])}}, output_dir=tmp_path,
                                  sampling_mode="random")
    assert output["movies_count"] == 0
    assert output["files"] == []
    assert old_movie.read_bytes() == b"previous run"


def test_subset_is_selected_before_sampling_and_video_encoding(tmp_path):
    calls = {}
    labels = np.array([0, 0, 0, 1, 1, 1])

    def extract(actual, **kwargs):
        np.testing.assert_array_equal(actual["recording"], labels)
        calls["window"] = kwargs
        return {0: [("recording", 0, 3)] * 4, 1: [("recording", 3, 6)] * 4}

    def sample(instances, count, **kwargs):
        calls["selected"] = instances
        assert count == 4
        return instances

    def encode(frames, path, **kwargs):
        calls["encoded"] = Path(path).name
        Path(path).write_bytes(b"tiny movie boundary fixture")

    backend = SimpleNamespace(
        filter_centroids_headings=lambda c, h, **kwargs: (c, h),
        check_video_paths=lambda *a: None, get_syllable_instances=extract,
        sample_instances=sample, grid_movie=lambda *a, **k: np.zeros((2, 2, 2, 3)),
        write_video_clip=encode,
    )
    record = {"syllable": labels, "centroid": np.zeros((6, 2)), "heading": np.zeros(6)}
    _generate_selected_grid_movies(backend, {"recording": record}, tmp_path,
        {"recording": "video.mp4"}, {"recording": np.zeros((6, 1, 2))}, None, None, None,
        10, 2, 2, .005, 3, False, {"mode": "random"}, [1], {"pre": .4, "post": .7, "window_size": 32})
    assert set(calls["selected"]) == {1}
    assert calls["window"]["pre"] == 4 and calls["window"]["post"] == 7
    assert calls["window"]["min_instances"] == 4
    assert calls["encoded"] == "syllable1.mp4"
    np.testing.assert_array_equal(record["syllable"], labels)
    assert not (tmp_path / "syllable0.mp4").exists()


def test_density_subset_keeps_other_syllables_in_reference_population(tmp_path):
    points = {0: np.array([0., .1, .2, 6., 8.]), 1: np.array([.1, .11, .12, 10., 12.])}
    instances = {key: [("recording", index, index + 1) for index in range(5)] for key in points}
    encoded = []
    fitted = []

    def sample(actual, count, **kwargs):
        fitted.append(set(actual))
        # Independent 1D nearest-neighbor reference: local/global density means
        # differ when the other syllable is retained. Target exemplar is index 3;
        # dropping the background makes every ratio 1 and selects index 0.
        background = np.concatenate([points[key] for key in actual])
        selected = {}
        for key in actual:
            local = np.sort(np.abs(points[key][:, None] - points[key]), axis=1)[:, :3].mean(axis=1)
            global_mean = np.sort(np.abs(points[key][:, None] - background), axis=1)[:, :3].mean(axis=1)
            exemplar = int(np.argmax(global_mean / local))
            selected[key] = [actual[key][exemplar]]
        return selected

    def frames(bouts, *args, **kwargs):
        encoded.append(bouts[0][1])
        return np.zeros((1, 2, 2, 3))

    backend = SimpleNamespace(
        filter_centroids_headings=lambda c, h, **kwargs: (c, h),
        check_video_paths=lambda *a: None, get_syllable_instances=lambda *a, **k: instances,
        sample_instances=sample, grid_movie=frames,
        write_video_clip=lambda frame_data, path, **k: Path(path).write_bytes(b"tiny boundary fixture"),
    )
    record = {"syllable": np.repeat([0, 1], 5), "centroid": np.zeros((10, 2)), "heading": np.zeros(10)}
    _generate_selected_grid_movies(backend, {"recording": record}, tmp_path,
        {"recording": "video.mp4"}, {"recording": np.zeros((10, 1, 2))}, None, None, None,
        10, 1, 1, 0., 1, False, {"mode": "density"}, [0], {"pre": 0., "post": 0., "window_size": 32})
    assert fitted == [{0, 1}]
    assert encoded == [3]
    assert (tmp_path / "syllable0.mp4").is_file()
    assert not (tmp_path / "syllable1.mp4").exists()


def test_density_interpolation_preserves_coordinates_for_other_artifacts(tmp_path):
    original = np.ones((6, 1, 2))
    original[1] = np.nan
    coordinates = {"recording": original}
    rendered_missing = []

    def sample(instances, count, coordinates, **kwargs):
        coordinates["recording"] = np.nan_to_num(coordinates["recording"])
        return instances

    def frames(*args, coordinates, **kwargs):
        rendered_missing.append(bool(np.isnan(coordinates["recording"]).any()))
        return np.zeros((1, 2, 2, 3))

    backend = SimpleNamespace(
        filter_centroids_headings=lambda c, h, **kwargs: (c, h),
        check_video_paths=lambda *a: None,
        get_syllable_instances=lambda *a, **k: {1: [("recording", 0, 6)]},
        sample_instances=sample, grid_movie=frames,
        write_video_clip=lambda frame_data, path, **k: Path(path).write_bytes(b"tiny boundary fixture"),
    )
    record = {"syllable": np.ones(6, dtype=int), "centroid": np.ones((6, 2)), "heading": np.zeros(6)}
    _generate_selected_grid_movies(backend, {"recording": record}, tmp_path,
        {"recording": "video.mp4"}, coordinates, None, None, None,
        1, 1, 1, 0., 1, False, {"mode": "density"}, [1], {"pre": 0., "post": 0., "window_size": 32})
    assert rendered_missing == [False]  # The grid retains the backend's interpolation.
    assert coordinates["recording"] is original
    assert np.isnan(coordinates["recording"][1]).all()  # Other artifacts retain raw missingness.


def fusion_fixture(tmp_path):
    top, bottom = tmp_path / "top", tmp_path / "bottom"
    top.mkdir()
    bottom.mkdir()
    return {"top_dir": str(top), "bot_dir": str(bottom), "output_dir": str(tmp_path / "fused")}


@pytest.mark.parametrize("name, static, expected", [
    ("continuous_calibrated_cohort.yaml", False, {"strict_calibration": True, "require_continuous_sync": True}),
    ("pilot_static.yaml", True, {"keypoint_mode": "21kp", "use_epochs": False}),
])
def test_checked_in_legacy_fusion_modes_keep_their_dispatch(repo_root, name, static, expected):
    cfg = yaml.safe_load((repo_root / "configs/fusion" / name).read_text())
    original = dict(cfg)
    actual_static, options = fusion_options(cfg)
    assert actual_static is static
    assert all(options[key] == value for key, value in expected.items())
    assert cfg == original


def test_fusion_yaml_options_and_cache_status_are_consumed(tmp_path, monkeypatch):
    cfg = fusion_fixture(tmp_path)
    cfg.update(mode="static_23kp", arena_width=100., use_epochs=True,
               truncate_aborted_tail=False, unmanifested_policy="rebuild")
    config = write_yaml(tmp_path / "fusion.yaml", cfg)
    monkeypatch.setattr(fuse, "find_matched_pairs", lambda **kwargs: [{"sid": "test"}])
    calls = []

    def backend(pair, **options):
        calls.append(options)
        return {"status": "CACHED", "cache_verification": "OK"}

    monkeypatch.setattr(fuse, "fuse_single_pair_static", backend)
    assert fuse.main(["--config", str(config), "--no-notify"]) == 0
    assert calls[0]["keypoint_mode"] == "23kp"
    assert calls[0]["arena_width"] == 100
    assert calls[0]["use_epochs"] is True
    assert calls[0]["truncate_aborted_tail"] is False
    assert calls[0]["unmanifested_policy"] == "rebuild"
    record = json.loads((tmp_path / "fused/run.json").read_text())
    assert record["status"] == "SUCCESS"
    assert (record["extra"]["success"], record["extra"]["failed"]) == (1, 0)
    assert record["config"]["use_epochs"] is True
    assert record["config_hash"] == compute_config_hash(record["config"])


@pytest.mark.parametrize("statuses, expected", [([], "FAILED"), (["ERROR"], "FAILED"), (["FUSED", "ERROR"], "PARTIAL_SUCCESS")])
def test_fusion_empty_and_failed_inputs_cannot_claim_success(tmp_path, monkeypatch, statuses, expected):
    cfg = fusion_fixture(tmp_path)
    config = write_yaml(tmp_path / "fusion.yaml", cfg)
    monkeypatch.setattr(fuse, "find_matched_pairs", lambda **kwargs: [{"sid": str(index)} for index in range(len(statuses))])
    monkeypatch.setattr(fuse, "fuse_single_pair_static", lambda pair, **kwargs: {"status": statuses[int(pair["sid"])]})
    assert fuse.main(["--config", str(config), "--no-notify"]) == 1
    assert json.loads((tmp_path / "fused/run.json").read_text())["status"] == expected
