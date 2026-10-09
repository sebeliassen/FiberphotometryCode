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
from fpcode.viz.grid_movies import _generate_selected_grid_movies, exact_grid_layout, generate_grid_movies
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


def fusion_fixture(tmp_path):
    top, bottom = tmp_path / "top", tmp_path / "bottom"
    top.mkdir()
    bottom.mkdir()
    return {"top_dir": str(top), "bot_dir": str(bottom), "output_dir": str(tmp_path / "fused")}


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
