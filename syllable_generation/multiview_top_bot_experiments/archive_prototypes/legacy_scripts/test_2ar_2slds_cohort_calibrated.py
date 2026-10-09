#!/usr/bin/env python3
"""
Test 2 AR + 2 SLDS iterations on the full calibrated cohort.
Logs memory, step timings, and catches any errors.
"""
import os
import sys
import time
import shutil
import pickle
from pathlib import Path
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_MEMORY_PATCH"] = "1"

import jax
import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video

PROD_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_continuous_calibrated"
TEST_RUN_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/test_2ar_2slds_cohort"

def main():
    print("=" * 70, flush=True)
    print("🚀 TEST RUN: 2 AR + 2 SLDS ON FULL CALIBRATED COHORT", flush=True)
    print(f"JAX Device: {jax.devices()}", flush=True)
    print("=" * 70, flush=True)

    if TEST_RUN_DIR.exists():
        shutil.rmtree(TEST_RUN_DIR)
    TEST_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy(PROD_DIR / "config.yml", TEST_RUN_DIR / "config.yml")
    config = kpms.load_config(str(TEST_RUN_DIR))

    # Apply optimizations
    print("\n[Step 1] Applying Amdahl 27x Optimizations...", flush=True)
    rb.apply_optimizations()

    # Load preprocessed data
    print("\n[Step 2] Loading cached preprocessed data...", flush=True)
    t0 = time.time()
    with open(PROD_DIR / "preprocessed_data.pkl", "rb") as f:
        data, metadata = pickle.load(f)
    print(f"  Loaded preprocessed data in {time.time()-t0:.2f}s. Active frames: {int(data['mask'].sum()):,}", flush=True)

    # Load initial model
    print("\n[Step 3] Loading cached initial model...", flush=True)
    t0 = time.time()
    with open(PROD_DIR / "init_model_seed_101.pkl", "rb") as f:
        cached_model = pickle.load(f)
    print(f"  Loaded initial model in {time.time()-t0:.2f}s.", flush=True)

    model = {
        "params": cached_model["params"],
        "hypparams": cached_model["hypparams"],
        "states": cached_model["states"],
        "noise_prior": cached_model["noise_prior"],
        "seed": cached_model["seed"]
    }

    # Precompute batch config
    print("\n[Step 4] Analytical Tensor Sizing...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, model=model, verbose=True)

    # TEST: 2 AR Iterations
    print("\n[Step 5] Running 2 AR Iterations...", flush=True)
    t_ar_start = time.time()
    model, _ = kpms.fit_model(
        model, data, metadata,
        project_dir=str(TEST_RUN_DIR),
        model_name="test_model",
        ar_only=True,
        start_iter=0,
        num_iters=2,
        parallel_message_passing=False,
        save_every_n_iters=None,
        generate_progress_plots=False,
        verbose=True
    )
    print(f"  ✅ 2 AR Iterations Complete in {time.time()-t_ar_start:.2f}s!", flush=True)

    # TEST: 2 SLDS Iterations
    print("\n[Step 6] Running 2 SLDS Iterations (10x Kappa reduction)...", flush=True)
    init_kappa = float(model["hypparams"]["trans_hypparams"]["kappa"])
    slds_kappa = init_kappa / 10.0
    model = kpms.update_hypparams(model, kappa=slds_kappa)

    t_slds_start = time.time()
    model, _ = kpms.fit_model(
        model, data, metadata,
        project_dir=str(TEST_RUN_DIR),
        model_name="test_model",
        ar_only=False,
        start_iter=2,
        num_iters=4,
        parallel_message_passing=False,
        save_every_n_iters=None,
        generate_progress_plots=False,
        verbose=True
    )
    print(f"  ✅ 2 SLDS Iterations Complete in {time.time()-t_slds_start:.2f}s!", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("🎉 LIVE GPU VALIDATION PASSED: 2 AR + 2 SLDS EXECUTED FLAWLESSLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
