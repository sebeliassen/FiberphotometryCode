#!/usr/bin/env python3
"""
diagnostics/profile_fit_model_gap.py

Times kpms.fit_model() step-by-step vs resample_model() on the exact same loaded model and data.
"""

import os
import sys
import time

sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode')
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import numpy as np

if not hasattr(np, "bool8"): np.bool8 = np.bool_

import keypoint_moseq as kpms
import utils.resample_batching as rb
from jax_moseq.models import keypoint_slds

PROJECT_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"

def sync(x):
    if hasattr(x, "block_until_ready"): x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for i in x: sync(i)
    elif isinstance(x, dict):
        for v in x.values(): sync(v)
    return x

def main():
    print("=== PROFILING FIT_MODEL GAP ===", flush=True)
    rb.apply_optimizations()

    print("Loading checkpoint...", flush=True)
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, "amph_prod_k200k_run", iteration=0)
    data["Y"] = np.asarray(data["Y"])

    # Test 1: Direct resample_model
    print("\n>>> Test 1: Direct keypoint_slds.resample_model(ar_only=True) (2 iters)...", flush=True)
    for i in range(2):
        t0 = time.perf_counter()
        model = keypoint_slds.resample_model(data, **model, ar_only=True)
        sync(model)
        print(f"  Direct Iteration {i+1}: {time.perf_counter()-t0:.2f} s", flush=True)

    # Test 2: fit_model
    print("\n>>> Test 2: kpms.fit_model(ar_only=True, num_iters=2, save_every_n_iters=None, generate_progress_plots=False)...", flush=True)
    t0 = time.perf_counter()
    model, _ = kpms.fit_model(
        model, data, metadata, project_dir=PROJECT_DIR,
        model_name="gap_test_run",
        ar_only=True,
        start_iter=0,
        num_iters=2,
        save_every_n_iters=None,
        generate_progress_plots=False,
        verbose=False,
    )
    sync(model)
    print(f"  fit_model (2 iters total): {time.perf_counter()-t0:.2f} s", flush=True)

if __name__ == "__main__":
    main()
