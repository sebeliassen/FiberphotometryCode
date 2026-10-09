#!/usr/bin/env python3
"""
diagnostics/profile_production_loop_gap.py

Isolates and directly measures the exact gap between:
Case A: Pure resample_model()
Case B: check_for_nans(model) overhead
Case C: _wrapped_resample(keypoint_slds.resample_model, ...)
Case D: fit_model() with NO checkpointing & NO progress plots
Case E: fit_model() with checkpointing & progress plots

All executed on the FULL 30-session AMPH dataset (562 sequences, 5.5M frames).
"""

import sys, os, time
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode')
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import numpy as np
if not hasattr(np, "bool8"): np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import keypoint_moseq as kpms
from keypoint_moseq.fitting import _wrapped_resample, check_for_nans
import utils.resample_batching as rb
from jax_moseq.models import keypoint_slds

PROJECT_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
SOURCE_MODEL = "amph_prod_k150k_run_01"

def sync(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for item in x: sync(item)
    elif isinstance(x, dict):
        for item in x.values(): sync(item)
    return x

def main():
    print("=" * 80)
    print("  ISOLATING THE PRODUCTION LOOP GAP (562 SEQUENCES, 5.5M FRAMES)")
    print("=" * 80, flush=True)

    rb.apply_optimizations()

    print("Loading production dataset (read-only)...", flush=True)
    t0 = time.time()
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, SOURCE_MODEL, iteration=0)
    print(f"Loaded in {time.time()-t0:.2f}s.\n", flush=True)

    # Warmup Gibbs kernels first
    print(">>> 1. Warming up Gibbs sampling kernels (1 SLDS iteration)...", flush=True)
    t0 = time.perf_counter()
    model = keypoint_slds.resample_model(data, **model, jitter=1e-3)
    sync(model)
    t_warmup = time.perf_counter() - t0
    print(f"--> Warmup finished in {t_warmup:.2f} s.\n", flush=True)

    # CASE A: Pure resample_model()
    print(">>> 2. Benchmarking Case A: Pure keypoint_slds.resample_model()...", flush=True)
    t0 = time.perf_counter()
    model = keypoint_slds.resample_model(data, **model, jitter=1e-3)
    sync(model)
    t_case_a = time.perf_counter() - t0
    print(f"--> Case A (Pure resample_model): {t_case_a:.2f} s\n", flush=True)

    # CASE B: check_for_nans(model)
    print(">>> 3. Benchmarking Case B: check_for_nans(model)...", flush=True)
    t0 = time.perf_counter()
    any_nans, nan_info, messages = check_for_nans(model)
    t_case_b = time.perf_counter() - t0
    print(f"--> Case B (check_for_nans): {t_case_b:.2f} s (any_nans={any_nans})\n", flush=True)

    # CASE C: _wrapped_resample
    print(">>> 4. Benchmarking Case C: _wrapped_resample(resample_model)...", flush=True)
    t0 = time.perf_counter()
    model = _wrapped_resample(keypoint_slds.resample_model, data, model, ar_only=False, jitter=1e-3)
    sync(model)
    t_case_c = time.perf_counter() - t0
    print(f"--> Case C (_wrapped_resample): {t_case_c:.2f} s\n", flush=True)

    # CASE D: fit_model() NO checkpointing, NO plots
    print(">>> 5. Benchmarking Case D: fit_model() (1 iter, save=None, plots=False)...", flush=True)
    t0 = time.perf_counter()
    model, _ = kpms.fit_model(
        model, data, metadata, project_dir=PROJECT_DIR,
        model_name="gap_test_d",
        num_iters=1, start_iter=0,
        ar_only=False,
        save_every_n_iters=None,
        generate_progress_plots=False,
        verbose=False
    )
    sync(model)
    t_case_d = time.perf_counter() - t0
    print(f"--> Case D (fit_model minimal): {t_case_d:.2f} s\n", flush=True)

    # CASE E: fit_model() WITH checkpoint save & progress plots
    print(">>> 6. Benchmarking Case E: fit_model() (1 iter, save=1, plots=True)...", flush=True)
    test_dir_e = os.path.join(PROJECT_DIR, "gap_test_e")
    if os.path.exists(test_dir_e):
        import shutil
        shutil.rmtree(test_dir_e)

    t0 = time.perf_counter()
    model, _ = kpms.fit_model(
        model, data, metadata, project_dir=PROJECT_DIR,
        model_name="gap_test_e",
        num_iters=1, start_iter=0,
        ar_only=False,
        save_every_n_iters=1,
        generate_progress_plots=True,
        verbose=False
    )
    sync(model)
    t_case_e = time.perf_counter() - t0
    print(f"--> Case E (fit_model with save+plots): {t_case_e:.2f} s\n", flush=True)

    # Cleanup test dir
    if os.path.exists(test_dir_e):
        import shutil
        shutil.rmtree(test_dir_e)

    print("=" * 80)
    print("  PRODUCTION LOOP GAP ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  * Case A (Pure Gibbs resample_model):       {t_case_a:6.2f} s")
    print(f"  * Case B (check_for_nans overhead):         {t_case_b:6.2f} s")
    print(f"  * Case C (_wrapped_resample):               {t_case_c:6.2f} s")
    print(f"  * Case D (fit_model minimal loop):          {t_case_d:6.2f} s")
    print(f"  * Case E (fit_model with HDF5 save & plots): {t_case_e:6.2f} s")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
