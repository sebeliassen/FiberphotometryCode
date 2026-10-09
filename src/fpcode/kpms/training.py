"""KPMS fitting orchestration; optional backend imports and patches are scoped."""

from __future__ import annotations

from contextlib import nullcontext
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union

import numpy as np

from fpcode.common.cache_manifest import atomic_write_json
from fpcode.common.logging import get_logger
from fpcode.common.paths import resolve_path
from fpcode.kpms.backend.patch_registry import temporary_patches
from fpcode.kpms.config import resolve_training_config

logger = get_logger("fpcode.kpms.fit")


def _load_training_backend(device):
    # Configuration is cheap and must precede importing or initializing JAX.
    from fpcode.kpms.backend.configuration import configure_jax_runtime
    configure_jax_runtime(preallocate=False, force_cpu=device == "cpu")
    import jax
    import keypoint_moseq as kpms
    from fpcode.kpms import runtime
    from fpcode.kpms.checkpoints import patched_load_checkpoint
    from fpcode.kpms.data import safe_fit_pca
    from fpcode.kpms.init_model import batched_init_model_by_video
    return SimpleNamespace(jax=jax, kpms=kpms, runtime=runtime, safe_fit_pca=safe_fit_pca,
                           init_batched=batched_init_model_by_video, checkpoint_loader=patched_load_checkpoint)


def _checkpoint_bindings(backend):
    kpms = backend.kpms
    bindings = [(kpms.io, "load_checkpoint", backend.checkpoint_loader),
                (kpms, "load_checkpoint", backend.checkpoint_loader)]
    if hasattr(kpms, "fitting") and hasattr(kpms.fitting, "load_checkpoint"):
        bindings.append((kpms.fitting, "load_checkpoint", backend.checkpoint_loader))
    return bindings


def _checkpoint_endpoint(path, expected):
    import h5py
    if not path.is_file():
        raise FileNotFoundError(f"Training returned a missing checkpoint: {path}")
    with h5py.File(path, "r") as handle:
        iterations = [int(value) for value in handle["model_snapshots"]]
    if not iterations or max(iterations) != expected:
        raise ValueError(f"Checkpoint does not contain the requested final iteration {expected}")


def _fit_phases(cfg, model_dir, seed, backend, data, metadata):
    kpms, runtime = backend.kpms, backend.runtime
    project_dir = model_dir.parent
    ar_iters, total = cfg["num_ar_iters"], cfg["total_iters"]
    checkpoint = model_dir / "checkpoint.h5"
    runtime.KPMSStateCache.clear()
    resumed_from = None
    if checkpoint.exists():
        model, data, metadata, current = kpms.load_checkpoint(str(project_dir), model_dir.name)
        if isinstance(current, bool) or int(current) != current or current < 0:
            raise ValueError(f"Invalid checkpoint iteration: {current}")
        current = int(current)
        if current >= total:
            raise ValueError(f"Checkpoint is already at iteration {current}; requested target is {total}. "
                             "Use a larger target or a fresh model directory.")
        resumed_from = current
        data["Y"] = np.asarray(data["Y"], dtype=np.float64)
    else:
        model, current = None, 0

    kpms_config = kpms.load_config(str(project_dir)) if (project_dir / "config.yml").exists() else {}
    kpms_config.update(cfg.get("kpms_options", {}))
    # Resolve PCA after checking the resume contract, to avoid unused artifacts.
    pca_path = resolve_path(cfg["pca_path"]) if cfg.get("pca_path") else project_dir / "pca.p"
    if pca_path.exists():
        with pca_path.open("rb") as handle:
            pca = pickle.load(handle)
    else:
        pca = backend.safe_fit_pca(data, project_dir)
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        with pca_path.open("wb") as handle:
            pickle.dump(pca, handle)
    if model is None:
        rng = backend.jax.random.PRNGKey(seed)
        if cfg["use_batched_init"]:
            model = backend.init_batched(data, pca, kpms_config, params_subset_size=5,
                                         batch_size=min(runtime.CONFIG.discrete_bs, 20), seed=rng)
        else:
            model = kpms.init_model(data, pca=pca, seed=rng,
                                    location_aware=cfg["location_aware"], **kpms_config)
        if cfg["init_kappa"] is not None:
            model = kpms.update_hypparams(model, kappa=cfg["init_kappa"])
    if cfg["use_memory_patch"]:
        runtime.precompute_optimal_batch_config(data, model=model, verbose=False)
    common = {"model_name": model_dir.name, "save_every_n_iters": cfg["save_every_n_iters"],
              "location_aware": cfg["location_aware"]}
    if current < ar_iters:
        model, _ = kpms.fit_model(model, data, metadata, str(project_dir), ar_only=True,
                                  start_iter=0 if current == 0 else current + 1,
                                  num_iters=ar_iters, **common)
        current = ar_iters
    if current < total:
        if cfg["slds_kappa"] is not None:
            model = kpms.update_hypparams(model, kappa=cfg["slds_kappa"])
        model, _ = kpms.fit_model(model, data, metadata, str(project_dir), ar_only=False,
                                  start_iter=current + 1, num_iters=total, **common)
    _checkpoint_endpoint(checkpoint, total)
    summary = {"model_name": model_dir.name, "seed": seed, "total_iters": total,
               "num_ar_iters": ar_iters, "num_slds_iters": cfg["num_slds_iters"],
               "resumed_from": resumed_from, "checkpoint_path": str(checkpoint)}
    atomic_write_json(model_dir / "model_summary.json", summary)
    return checkpoint


def fit_model(cfg: Dict[str, Any], model_dir: Union[Path, str], seed: int = 101,
              device: str = "gpu") -> Path:
    """Fit AR then SLDS phases, restoring backend patches on every exit.

    Existing checkpoints must be below the requested endpoint. Already-complete
    or later checkpoints fail explicitly without claiming a new training run.
    Disabling memory patches means this call installs none; callers' existing
    patches are retained. Backend scopes are process-global and must be serialized.
    """
    if device not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be cpu, gpu or cuda")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cfg = resolve_training_config(cfg)
    model_dir = Path(model_dir).resolve()
    data_path = resolve_path(cfg["data_path"]) if cfg.get("data_path") else model_dir.parent / "preprocessed_data.pkl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    with data_path.open("rb") as handle:
        loaded = pickle.load(handle)
    if isinstance(loaded, tuple):
        data, metadata = loaded[0], loaded[1] if len(loaded) > 1 else None
    elif isinstance(loaded, dict):
        data, metadata = loaded.get("data", loaded), loaded.get("metadata")
    else:
        raise ValueError(f"Unrecognized training data format in {data_path}")
    if not isinstance(data, dict) or "Y" not in data:
        raise ValueError("Preprocessed training data must contain observation array Y")
    data["Y"] = np.asarray(data["Y"], dtype=np.float64)
    backend = _load_training_backend(device)
    model_dir.mkdir(parents=True, exist_ok=True)
    # Captures each loader namespace's own prior value, including nested scopes.
    try:
        with temporary_patches(_checkpoint_bindings(backend)):
            scope = backend.runtime.kpms_optimizations() if cfg["use_memory_patch"] else nullcontext()
            with scope:
                return _fit_phases(cfg, model_dir, seed, backend, data, metadata)
    finally:
        backend.runtime.KPMSStateCache.clear()


def run_seed_queue(seeds: List[int], cfg: Dict[str, Any], base_dir: Union[Path, str],
                   device: str = "gpu") -> List[Path]:
    """Run seeds sequentially; every fitting call restores its own patch scope."""
    return [fit_model(cfg, Path(base_dir) / f"model_seed{seed}", seed=seed, device=device) for seed in seeds]
