#!/usr/bin/env python3
"""
test_2ar_2slds_tandem_full.py
=============================================================================
Runs 2 AR + 2 SLDS iterations on the full 23-keypoint continuous tandem dataset
(416,032 frames across 42 segmented sequences, 23 keypoints @ 60 FPS) to prove
complete end-to-end execution capability on the live GPU.

Operates in an isolated test directory to guarantee zero interference with
existing production checkpoints or pickles.
=============================================================================
"""

import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# GPU Environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_MEMORY_PATCH"] = "1"

import jax
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video

TEST_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/test_2ar_2slds_verification"
TEST_DATA_DIR = TEST_DIR / "data"
PROD_DATA_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/kpms_tandem_production/data"

def main():
    print("=" * 80)
    print("🚀 LIVE GPU VALIDATION: 2 AR + 2 SLDS ITERATIONS ON FULL TANDEM DATASET")
    print(f"JAX Devices: {jax.devices()}")
    print(f"Working Directory: {TEST_DIR}")
    print("=" * 80)

    # Clean test directory
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Ingest full tandem production data (2 full continuous sessions = 416,032 frames)
    h5_files = sorted(PROD_DATA_DIR.glob("*.h5"))
    print(f"\n[Step 1/5] Ingesting {len(h5_files)} tandem H5 files from {PROD_DATA_DIR}...")
    for f in h5_files:
        dst = TEST_DATA_DIR / f.name
        dst.symlink_to(f)
        print(f"  Linked: {f.name}")

    # Copy project config
    config_source = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project/config.yml"
    shutil.copy(config_source, TEST_DIR / "config.yml")
    config = kpms.load_config(str(TEST_DIR))
    config["fps"] = 60
    init_kappa = 7.5e13
    slds_kappa = 7.5e12
    config["trans_hypparams"]["kappa"] = init_kappa
    kpms.update_config(str(TEST_DIR), **config)

    # Load and format data
    print("\n[Step 2/5] Loading 23-keypoint tracking data and formatting segments...")
    coordinates, confidences, bodyparts = kpms.load_keypoints(str(TEST_DATA_DIR), format="deeplabcut")
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    data = convert_data_precision(data, np.float64)
    data["Y"] = np.asarray(data["Y"])
    
    total_frames = sum(len(c) for c in coordinates.values())
    n_segs, seg_len, n_kps, d = data["Y"].shape
    print(f"  Total raw frames loaded: {total_frames:,}")
    print(f"  Batched Y tensor: {n_segs} sequences x {seg_len} frames x {n_kps} keypoints x {d}D")
    assert n_kps == 23, f"Expected 23 keypoints, got {n_kps}"

    # Load or fit PCA
    print("\n[Step 3/5] Fitting PCA (D=10)...")
    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, str(TEST_DIR))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Top 10 PCs Cumulative Variance Explained: {cum_var[-1]*100:.2f}%")

    # Analytical memory sizing
    print("\n[Step 4/5] Preemptive Analytical Tensor Sizing & Caching...")
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=True)

    # Initialize model
    print("\nInitializing fresh AR-HMM model with batched init...")
    init_bs = min(rb.CONFIG.discrete_bs, 20)
    rng_key = jax.random.PRNGKey(42)
    model = batched_init_model_by_video(
        data, pca, config,
        params_subset_size=5, batch_size=init_bs, seed=rng_key
    )
    model = kpms.update_hypparams(model, kappa=float(init_kappa))
    print("Model initialized successfully.")

    # Execute 2 AR iterations
    print("\n" + "=" * 80)
    print("🔥 PHASE 1: EXECUTING 2 AR-HMM ITERATIONS...")
    print("=" * 80)
    t0_ar = time.time()
    model, _ = kpms.fit_model(
        model, data, metadata,
        project_dir=str(TEST_DIR),
        model_name="test_model",
        ar_only=True,
        start_iter=0,
        num_iters=2,
        parallel_message_passing=False,
        save_every_n_iters=1,
        generate_progress_plots=False,
        verbose=True
    )
    t_ar = time.time() - t0_ar
    print(f"  >>> Phase 1 (2 AR Iterations) completed in {t_ar:.2f}s ({t_ar/2:.2f}s/iter)")

    # Execute 2 SLDS iterations
    print("\n" + "=" * 80)
    print("🔥 PHASE 2: EXECUTING 2 FULL SLDS ITERATIONS (Banded Cholesky + GPU FFBS)...")
    print("=" * 80)
    # Transition kappa to SLDS kappa
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    
    t0_slds = time.time()
    model, _ = kpms.fit_model(
        model, data, metadata,
        project_dir=str(TEST_DIR),
        model_name="test_model",
        ar_only=False,
        start_iter=2,
        num_iters=4,
        parallel_message_passing=False,
        save_every_n_iters=1,
        generate_progress_plots=False,
        verbose=True
    )
    t_slds = time.time() - t0_slds
    print(f"  >>> Phase 2 (2 SLDS Iterations) completed in {t_slds:.2f}s ({t_slds/2:.2f}s/iter)")

    # Step 5: Verification & Diagnostics
    print("\n[Step 5/5] Auditing State Consistency and Syllables...")
    x = np.asarray(model["states"]["x"])
    z = np.asarray(model["states"]["z"])
    mask = np.asarray(data["mask"])[:, 3:]

    # Check for NaNs
    for name, arr in [("x (latents)", x), ("z (syllables)", z), ("Ab (AR)", model["params"]["Ab"]), ("Q (noise)", model["params"]["Q"])]:
        nan_count = np.sum(np.isnan(arr))
        print(f"  NaN check for {name}: {nan_count} NaNs")
        assert nan_count == 0, f"Found {nan_count} NaNs in {name}!"

    # Syllables
    active_sylls = len(np.unique(z[mask > 0]))
    durations = kpms.viz.get_durations(z, mask)
    print(f"  Active Syllables: {active_sylls}")
    print(f"  Total Syllable Bouts: {len(durations):,}")
    print(f"  Median Bout Duration: {np.median(durations):.1f} frames ({(np.median(durations)/60)*1000:.1f} ms)")
    print(f"  Mean Bout Duration:   {np.mean(durations):.1f} frames ({(np.mean(durations)/60)*1000:.1f} ms)")

    print("\n" + "=" * 80)
    print("🏆 SUCCESS! LIVE EXECUTION COMPLETED WITHOUT ERRORS!")
    print(f"   Total 4 iterations (2 AR + 2 SLDS) on 416,032 frames ran in {t_ar + t_slds:.2f} seconds.")
    print("=" * 80)

if __name__ == "__main__":
    main()
