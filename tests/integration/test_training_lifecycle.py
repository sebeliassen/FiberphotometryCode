"""Exercise real fitting orchestration with tiny external-backend substitutes."""

import importlib
import json
import pickle
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fpcode.kpms.backend.patch_registry import temporary_patches
from fpcode.kpms.fit import fit_model

training = importlib.import_module(fit_model.__module__)


def fake_backend(tmp_path, monkeypatch, *, fail=None, resume=None, stop=None):
    model_dir = tmp_path / "model"
    data = {"Y": np.zeros((1, 3, 2))}
    with (tmp_path / "preprocessed_data.pkl").open("wb") as handle:
        pickle.dump({"data": data, "metadata": ("recording",)}, handle)
    if resume is None:
        with (tmp_path / "pca.p").open("wb") as handle:
            pickle.dump({"tiny": True}, handle)
    else:
        model_dir.mkdir()
        with h5py.File(model_dir / "checkpoint.h5", "w") as handle:
            handle.create_dataset(f"model_snapshots/{resume}/completed_updates", data=resume)
    previous_root, previous_io, previous_fitting, entry_kernel, replacement = [object() for _ in range(5)]
    target = SimpleNamespace(kernel=entry_kernel)
    io = SimpleNamespace(load_checkpoint=previous_io)
    fitting = SimpleNamespace(load_checkpoint=previous_fitting)
    calls, cache_clears, updates = [], [], []
    loader = lambda *a: ({"completed_updates": resume}, data, None, resume)

    class StopResampling(Exception):
        pass

    def resample(resample_func, observations, model, **kwargs):
        phase = "ar" if calls[-1]["ar_only"] else "slds"
        if stop is not None and phase == stop[0] and updates.count(phase) >= stop[1]:
            raise StopResampling()
        updates.append(phase)
        return dict(model, completed_updates=model["completed_updates"] + 1)

    fitting._wrapped_resample = resample
    fitting.completed_updates = updates
    fitting.original_resample = resample

    @contextmanager
    def optimizations():
        with temporary_patches([(target, "kernel", replacement)]):
            yield

    def fit(model, observations, metadata, project_dir, **kwargs):
        assert kpms.load_checkpoint is loader
        assert io.load_checkpoint is loader and fitting.load_checkpoint is loader
        assert target.kernel is replacement
        calls.append(kwargs)
        if fail == ("ar" if kwargs["ar_only"] else "slds"):
            raise RuntimeError("synthetic backend failure")
        # Mirror installed KPMS: initialize snapshot at start_iter, sample the
        # inclusive range, swallow StopResampling, and save only > start_iter.
        with h5py.File(model_dir / "checkpoint.h5", "a") as handle:
            if "model_snapshots" not in handle:
                handle.create_dataset(f"model_snapshots/{kwargs['start_iter']}/completed_updates",
                                      data=model["completed_updates"])
            else:
                for key in list(handle["model_snapshots"]):
                    if int(key) > kwargs["start_iter"]:
                        del handle[f"model_snapshots/{key}"]
            for iteration in range(kwargs["start_iter"], kwargs["num_iters"] + 1):
                try:
                    model = fitting._wrapped_resample(None, observations, model)
                except StopResampling:
                    break
                if iteration > kwargs["start_iter"] and (
                    iteration == kwargs["num_iters"] or iteration % kwargs["save_every_n_iters"] == 0
                ):
                    handle.create_dataset(f"model_snapshots/{iteration}/completed_updates",
                                          data=model["completed_updates"])
        return model, None

    kpms = SimpleNamespace(io=io, fitting=fitting, load_checkpoint=previous_root,
        fit_model=fit, init_model=lambda *a, **k: {"completed_updates": 0},
        update_hypparams=lambda model, **kwargs: model)
    runtime = SimpleNamespace(CONFIG=SimpleNamespace(discrete_bs=10),
        KPMSStateCache=SimpleNamespace(clear=lambda: cache_clears.append(True)),
        kpms_optimizations=optimizations, precompute_optimal_batch_config=lambda *a, **k: None)
    backend = SimpleNamespace(kpms=kpms, runtime=runtime, checkpoint_loader=loader,
        jax=SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: seed)),
        init_batched=lambda *a, **k: {"completed_updates": 0},
        safe_fit_pca=lambda *a: pytest.fail("Unexpected PCA fitting"))
    monkeypatch.setattr(training, "_load_training_backend", lambda device: backend)
    return model_dir, backend, target, (previous_root, previous_io, previous_fitting, entry_kernel), calls, cache_clears


@pytest.mark.parametrize("fail", [None, "ar", "slds"])
def test_training_phase_parameters_and_patch_cleanup(tmp_path, monkeypatch, fail):
    model_dir, backend, target, originals, calls, clears = fake_backend(tmp_path, monkeypatch, fail=fail)
    cfg = {"hyperparameters": {"num_ar_iters": 2, "num_slds_iters": 3, "save_every_n_iters": 1}}
    if fail is not None:
        with pytest.raises(RuntimeError, match="synthetic backend failure"):
            training.fit_model(cfg, model_dir, seed=17, device="cpu")
        assert not (model_dir / "model_summary.json").exists()
    else:
        assert training.fit_model(cfg, model_dir, seed=17, device="cpu") == model_dir / "checkpoint.h5"
    assert backend.kpms.load_checkpoint is originals[0]
    assert backend.kpms.io.load_checkpoint is originals[1]
    assert backend.kpms.fitting.load_checkpoint is originals[2]
    assert target.kernel is originals[3]
    assert clears
    assert backend.kpms.fitting._wrapped_resample is backend.kpms.fitting.original_resample
    assert calls[0]["ar_only"] is True and calls[0]["num_iters"] == 2 and calls[0]["start_iter"] == 0
    if len(calls) > 1:
        assert calls[1]["ar_only"] is False and calls[1]["num_iters"] == 5 and calls[1]["start_iter"] == 2
    if fail is None:
        assert backend.kpms.fitting.completed_updates == ["ar", "ar", "slds", "slds", "slds"]
        with h5py.File(model_dir / "checkpoint.h5", "r") as handle:
            assert int(handle["model_snapshots/5/completed_updates"][()]) == 5


@pytest.mark.parametrize("resume", [5, 6])
def test_already_complete_or_later_resume_is_explicit_failure(tmp_path, monkeypatch, resume):
    model_dir, backend, target, originals, calls, clears = fake_backend(tmp_path, monkeypatch, resume=resume)
    with pytest.raises(ValueError, match="already at iteration"):
        training.fit_model({"num_ar_iters": 2, "num_slds_iters": 3}, model_dir, device="cpu")
    assert not calls and not (tmp_path / "pca.p").exists()
    assert not (model_dir / "model_summary.json").exists()
    assert backend.kpms.load_checkpoint is originals[0]
    assert target.kernel is originals[3]
    assert clears


def test_invalid_device_fails_before_backend_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_load_training_backend", lambda *a: pytest.fail("Backend was loaded"))
    with pytest.raises(ValueError, match="device"):
        training.fit_model({}, tmp_path / "model", device="invalid")


def test_runtime_environment_is_configured_before_jax_import(monkeypatch):
    import builtins
    import os
    original_import = builtins.__import__
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before")
    monkeypatch.setenv("JAX_PLATFORMS", "before")

    def import_boundary(name, *args, **kwargs):
        if name == "jax":
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
            assert os.environ["JAX_PLATFORMS"] == "cpu"
            raise RuntimeError("stopped before real JAX import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_boundary)
    with pytest.raises(RuntimeError, match="stopped before real JAX import"):
        training._load_training_backend("cpu")


@pytest.mark.parametrize("ar_iters,slds_iters,resume,expected", [
    (2, 1, None, ["ar", "ar", "slds"]),
    (0, 1, None, ["slds"]),
    (1, 0, None, ["ar"]),
    (2, 3, 4, ["slds"]),
    (2, 3, 1, ["ar", "slds", "slds", "slds"]),
])
def test_one_update_phases_and_resume_persist_actual_completed_updates(
    tmp_path, monkeypatch, ar_iters, slds_iters, resume, expected,
):
    model_dir, backend, target, originals, calls, clears = fake_backend(tmp_path, monkeypatch, resume=resume)
    total = ar_iters + slds_iters
    training.fit_model({"num_ar_iters": ar_iters, "num_slds_iters": slds_iters}, model_dir, device="cpu")
    assert backend.kpms.fitting.completed_updates == expected
    with h5py.File(model_dir / "checkpoint.h5", "r") as handle:
        assert int(handle[f"model_snapshots/{total}/completed_updates"][()]) == total
    summary = json.loads((model_dir / "model_summary.json").read_text())
    assert summary["resumed_from"] == resume and summary["total_iters"] == total
    if resume is not None:
        assert not (tmp_path / "pca.p").exists()
    assert backend.kpms.fitting._wrapped_resample is backend.kpms.fitting.original_resample
    assert backend.kpms.load_checkpoint is originals[0] and target.kernel is originals[3]


@pytest.mark.parametrize("stop", [("ar", 0), ("ar", 1), ("slds", 0), ("slds", 1)])
def test_silent_backend_early_stop_cannot_report_success(tmp_path, monkeypatch, stop):
    model_dir, backend, target, originals, calls, clears = fake_backend(tmp_path, monkeypatch, stop=stop)
    with pytest.raises(RuntimeError, match="Training phase stopped"):
        training.fit_model({"num_ar_iters": 2, "num_slds_iters": 3}, model_dir, device="cpu")
    assert not (model_dir / "model_summary.json").exists()
    if stop[0] == "ar":
        assert len(calls) == 1
    assert backend.kpms.fitting._wrapped_resample is backend.kpms.fitting.original_resample
    assert backend.kpms.load_checkpoint is originals[0] and target.kernel is originals[3]


@pytest.mark.parametrize("batched", [False, True])
def test_training_forwards_location_aware_to_initialization(tmp_path, monkeypatch, batched):
    model_dir, backend, target, originals, calls, clears = fake_backend(tmp_path, monkeypatch)
    initialization = []

    def init(*args, **kwargs):
        initialization.append(kwargs)
        return {"completed_updates": 0}

    backend.init_batched = backend.kpms.init_model = init
    training.fit_model({"num_ar_iters": 1, "num_slds_iters": 1,
                        "location_aware": True, "use_batched_init": batched}, model_dir, device="cpu")
    assert initialization[0]["location_aware"] is True
    assert all(call["location_aware"] is True for call in calls)


def test_batched_initialization_uses_location_aware_variant_for_every_batch(monkeypatch):
    from fpcode.kpms.init_model import batched_init_model_by_video
    calls = []

    def init(data, **kwargs):
        calls.append(kwargs)
        n = len(data["Y"])
        return {"seed": kwargs["seed"], "states": {"x": np.zeros((n, 3, 2))},
                "params": {"delta_v": "location-aware"}, "hypparams": {}, "noise_prior": 1.0}

    jax = SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: seed,
                                                 split=lambda seed: (seed + 1, seed + 2)))
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "keypoint_moseq", SimpleNamespace(init_model=init))
    data = {"Y": np.zeros((3, 3, 2)), "mask": np.ones((3, 3))}
    result = batched_init_model_by_video(data, {}, {}, batch_size=2, seed=17, location_aware=True)
    assert len(calls) == 3 and all(call["location_aware"] is True for call in calls)
    assert result["states"]["x"].shape == (3, 3, 2)
    assert result["params"]["delta_v"] == "location-aware"


@pytest.mark.parametrize("with_conf,explicit_prior", [(False, False), (True, False), (True, True)])
def test_batched_initialization_assembles_noise_priors_and_retains_global_parameters(
    monkeypatch, with_conf, explicit_prior,
):
    from fpcode.kpms.init_model import batched_init_model_by_video
    calls = []

    def init(data, **kwargs):
        assert "conf" not in data or data["conf"] is not None
        n = len(data["Y"])
        # A fresh parameter estimate differs by batch length; reused parameters
        # must keep every batch's states in the subset model's chosen basis.
        params = kwargs.get("params", {"Cd": n})
        hypparams = kwargs.get("hypparams", {"variant": "selected"})
        noise = kwargs.get("noise_prior", data["conf"] * 2 if "conf" in data else 1.0)
        calls.append((data, kwargs))
        return {"seed": kwargs["seed"], "states": {"x": np.full((n, 3, 2), params["Cd"])},
                "params": params, "hypparams": hypparams, "noise_prior": noise}

    jax = SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: seed,
                                                 split=lambda seed: (seed + 1, seed + 2)))
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "keypoint_moseq", SimpleNamespace(init_model=init))
    data = {"Y": np.zeros((7, 3, 2, 2)), "mask": np.ones((7, 3))}
    confidence = np.arange(1, 8)[:, None, None] * np.ones((7, 3, 2))
    if with_conf:
        data["conf"] = confidence
    config = {"noise_prior": confidence * 3} if explicit_prior else {}
    result = batched_init_model_by_video(data, {}, config, params_subset_size=2, batch_size=3, seed=17)
    assert len(calls) == 4
    for batch_data, options in calls[1:]:
        assert options["params"] is result["params"]
        assert options["hypparams"] is result["hypparams"]
    np.testing.assert_array_equal(result["states"]["x"], np.full((7, 3, 2), 2))
    if with_conf:
        np.testing.assert_array_equal(result["noise_prior"], confidence * (3 if explicit_prior else 2))
        np.testing.assert_array_equal(data["conf"], confidence)
    else:
        assert result["noise_prior"] == 1.0
        assert all("conf" not in batch_data for batch_data, _ in calls)


def test_missing_backend_iteration_hook_fails_explicitly(tmp_path, monkeypatch):
    model_dir, backend, target, originals, calls, clears = fake_backend(tmp_path, monkeypatch)
    del backend.kpms.fitting._wrapped_resample
    with pytest.raises(RuntimeError, match="must expose callable _wrapped_resample"):
        training.fit_model({"num_ar_iters": 1, "num_slds_iters": 1}, model_dir, device="cpu")
    assert not calls
    assert backend.kpms.load_checkpoint is originals[0] and target.kernel is originals[3]
