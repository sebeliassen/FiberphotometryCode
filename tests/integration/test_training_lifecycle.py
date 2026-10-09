"""Exercise real fitting orchestration with tiny external-backend substitutes."""

import importlib
import pickle
from contextlib import contextmanager
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fpcode.kpms.fit import fit_model
from fpcode.kpms.backend.patch_registry import temporary_patches

training = importlib.import_module(fit_model.__module__)


def fake_backend(tmp_path, monkeypatch, *, fail=None, resume=None):
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
            handle.create_group(f"model_snapshots/{resume}")
    previous_root, previous_io, previous_fitting, entry_kernel, replacement = [object() for _ in range(5)]
    target = SimpleNamespace(kernel=entry_kernel)
    io = SimpleNamespace(load_checkpoint=previous_io)
    fitting = SimpleNamespace(load_checkpoint=previous_fitting)
    calls, cache_clears = [], []
    loader = lambda *a: ({}, data, None, resume)

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
        with h5py.File(model_dir / "checkpoint.h5", "a") as handle:
            handle.create_group(f"model_snapshots/{kwargs['num_iters']}")
        return model, None

    kpms = SimpleNamespace(io=io, fitting=fitting, load_checkpoint=previous_root,
        fit_model=fit, init_model=lambda *a, **k: {}, update_hypparams=lambda model, **kwargs: model)
    runtime = SimpleNamespace(CONFIG=SimpleNamespace(discrete_bs=10),
        KPMSStateCache=SimpleNamespace(clear=lambda: cache_clears.append(True)),
        kpms_optimizations=optimizations, precompute_optimal_batch_config=lambda *a, **k: None)
    backend = SimpleNamespace(kpms=kpms, runtime=runtime, checkpoint_loader=loader,
        jax=SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: seed)),
        init_batched=lambda *a, **k: {}, safe_fit_pca=lambda *a: pytest.fail("Unexpected PCA fitting"))
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
    assert calls[0]["ar_only"] is True and calls[0]["num_iters"] == 2 and calls[0]["start_iter"] == 0
    if len(calls) > 1:
        assert calls[1]["ar_only"] is False and calls[1]["num_iters"] == 5 and calls[1]["start_iter"] == 3


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
