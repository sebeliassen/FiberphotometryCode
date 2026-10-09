"""
fpcode.kpms.fit
===============
Model fitting engine, seed queues, and multi-phase AR / SLDS training for Keypoint-MoSeq models.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import jax
import keypoint_moseq as kpms
import numpy as np

from fpcode.common.logging import get_logger
from fpcode.kpms.checkpoints import apply_kpms_checkpoint_patch
from fpcode.kpms.data import safe_fit_pca
from fpcode.kpms.init_model import batched_init_model_by_video
from fpcode.kpms.runtime import (
    CONFIG as RB_CONFIG,
)
from fpcode.kpms.runtime import (
    KPMSStateCache,
    apply_optimizations,
    configure_jax_runtime,
    precompute_optimal_batch_config,
    restore_original_kernels,
)

logger = get_logger("fpcode.kpms.fit")


def fit_model(
    cfg: Dict[str, Any],
    model_dir: Union[Path, str],
    seed: int = 101,
    device: str = "gpu",
) -> Path:
    """
    Fits a Keypoint-MoSeq model driven by a declarative configuration.

    Parameters
    ----------
    cfg : dict
        Training configuration dictionary.
    model_dir : Path or str
        Destination directory for checkpoints, progress plots, and summaries.
    seed : int, default=101
        Random seed for model initialization and JAX PRNG.
    device : {"gpu", "cuda", "cpu"}, default="gpu"
        Execution device.

    Returns
    -------
    Path
        Path to the output checkpoint.h5.
    """
    force_cpu = device.lower() == "cpu"
    configure_jax_runtime(preallocate=False, force_cpu=force_cpu)

    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    project_dir = model_dir.parent

    # Apply transparent checkpoint patch
    apply_kpms_checkpoint_patch()

    use_memory_patch = cfg.get("use_memory_patch", True)
    if use_memory_patch:
        apply_optimizations()
    else:
        restore_original_kernels()

    num_ar_iters = int(cfg.get("num_ar_iters", 50))
    num_slds_iters = int(cfg.get("num_slds_iters", 200))
    total_iters = num_ar_iters + num_slds_iters
    save_every_n = int(cfg.get("save_every_n_iters", 25))
    use_batched_init = bool(cfg.get("use_batched_init", True))
    location_aware = bool(cfg.get("location_aware", False))

    init_kappa = cfg.get("init_kappa", cfg.get("ar_kappa", 1.0e6))
    slds_kappa = cfg.get("slds_kappa", 1.5e15)

    kpms_config = kpms.load_config(str(project_dir)) if (project_dir / "config.yml").exists() else {}
    kpms_config.update(cfg.get("kpms_options", {}))

    # 1. Resolve observation data
    data_path = cfg.get("data_path") or (project_dir / "preprocessed_data.pkl")
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    with open(data_path, "rb") as f:
        loaded_data = pickle.load(f)

    if isinstance(loaded_data, tuple):
        data = loaded_data[0]
        metadata = loaded_data[1] if len(loaded_data) > 1 else None
    elif isinstance(loaded_data, dict):
        data = loaded_data.get("data", loaded_data)
        metadata = loaded_data.get("metadata", None)
    else:
        raise ValueError(f"Unrecognized data format in {data_path}")

    data["Y"] = np.asarray(data["Y"], dtype=np.float64)

    # 2. Resolve PCA
    pca_file = cfg.get("pca_path") or (project_dir / "pca.p")
    pca_file = Path(pca_file)
    if pca_file.exists():
        logger.info(f"Loading PCA from {pca_file}")
        with open(pca_file, "rb") as pf:
            pca = pickle.load(pf)
    else:
        logger.info(f"Fitting PCA and saving to {pca_file}...")
        pca = safe_fit_pca(data, project_dir)
        with open(pca_file, "wb") as pf:
            pickle.dump(pca, pf)

    # 3. Check for existing checkpoint (resume vs init)
    ckpt_path = model_dir / "checkpoint.h5"
    KPMSStateCache.clear()

    if ckpt_path.exists():
        logger.info(f"Resuming existing checkpoint at {ckpt_path}...")
        model, data, metadata, current_iter = kpms.load_checkpoint(str(project_dir), model_dir.name)
        data["Y"] = np.asarray(data["Y"], dtype=np.float64)
    else:
        logger.info(f"Initializing model with seed {seed}...")
        rng_key = jax.random.PRNGKey(seed)
        if use_batched_init:
            init_bs = min(RB_CONFIG.discrete_bs, 20)
            model = batched_init_model_by_video(
                data, pca, kpms_config,
                params_subset_size=5, batch_size=init_bs, seed=rng_key,
            )
        else:
            model = kpms.init_model(data, pca=pca, seed=rng_key, location_aware=location_aware, **kpms_config)

        if init_kappa is not None:
            model = kpms.update_hypparams(model, kappa=float(init_kappa))
        current_iter = 0

    if use_memory_patch:
        precompute_optimal_batch_config(data, model=model, verbose=False)

    # 4. Phase 1: AR iterations
    if current_iter < num_ar_iters:
        logger.info(f"Phase 1: AR fit from iteration {current_iter} to {num_ar_iters}...")
        model, _ = kpms.fit_model(
            model, data, metadata, str(project_dir),
            model_name=model_dir.name,
            ar_only=True,
            start_iter=0 if current_iter == 0 else current_iter + 1,
            num_iters=num_ar_iters,
            save_every_n_iters=save_every_n,
            location_aware=location_aware,
        )
        current_iter = num_ar_iters

    # 5. Phase 2: SLDS iterations
    if current_iter < total_iters:
        logger.info(f"Phase 2: SLDS fit from iteration {current_iter} to {total_iters}...")
        if slds_kappa is not None:
            model = kpms.update_hypparams(model, kappa=float(slds_kappa))

        model, _ = kpms.fit_model(
            model, data, metadata, str(project_dir),
            model_name=model_dir.name,
            ar_only=False,
            start_iter=current_iter + 1,
            num_iters=total_iters,
            save_every_n_iters=save_every_n,
            location_aware=location_aware,
        )
        current_iter = total_iters

    # 6. Write model summary
    summary = {
        "model_name": model_dir.name,
        "seed": seed,
        "total_iters": total_iters,
        "num_ar_iters": num_ar_iters,
        "num_slds_iters": num_slds_iters,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint_path": str(ckpt_path),
    }
    with open(model_dir / "model_summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)

    logger.info(f"Training successfully finished for {model_dir.name} ({total_iters} iterations).")
    return ckpt_path


def run_seed_queue(
    seeds: List[int],
    cfg: Dict[str, Any],
    base_dir: Union[Path, str],
    device: str = "gpu",
) -> List[Path]:
    """Runs a sequential queue of training seeds."""
    base_dir = Path(base_dir)
    results = []
    for s in seeds:
        m_dir = base_dir / f"model_seed{s}"
        res = fit_model(cfg, m_dir, seed=s, device=device)
        results.append(res)
    return results
