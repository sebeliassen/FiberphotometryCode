#!/usr/bin/env python3
"""
diagnostics/debug_ar_iteration_speed.py

Instruments every step of an AR iteration on the real 562-sequence AMPH dataset
to find out EXACTLY where the seconds are spent.
"""

import os, sys, time
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
import utils.resample_batching as rb
from jax_moseq.models import arhmm, keypoint_slds
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs

PROJECT_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
CKPT = os.path.join(PROJECT_DIR, "amph_prod_k200k_run", "checkpoint.h5")

def sync(x):
    if hasattr(x, "block_until_ready"): x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for i in x: sync(i)
    elif isinstance(x, dict):
        for v in x.values(): sync(v)
    return x

def main():
    print("=== DEBUGGING AR ITERATION SPEED (FULL 562 SEQUENCES) ===")
    rb.apply_optimizations()
    
    print("Loading checkpoint...")
    import h5py
    # Load iteration 0
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, "amph_prod_k200k_run", iteration=0)
    data["Y"] = np.asarray(data["Y"])
    N = data["mask"].shape[0]
    T = data["mask"].shape[1]
    print(f"Loaded: N={N}, T={T}")
    
    # 1. Warm up
    print("\n--- Warming up AR kernels (1 iter) ---")
    t0 = time.perf_counter()
    model = keypoint_slds.resample_model(data, **model, ar_only=True)
    sync(model)
    print(f"Warmup done in {time.perf_counter()-t0:.2f}s")
    
    # 2. Granular breakdown of 1 AR iteration
    print("\n--- Granular Breakdown of Warmed AR Iteration ---")
    seed = model["seed"]
    states = model["states"].copy()
    params = model["params"].copy()
    hypparams = model["hypparams"]
    
    # Transitions
    t0 = time.perf_counter()
    betas, pi = arhmm_gibbs.resample_hdp_transitions(
        seed, **data, **states, **params, **hypparams["trans_hypparams"]
    )
    sync(pi)
    t_trans = time.perf_counter() - t0
    print(f"1. resample_hdp_transitions: {t_trans:6.2f} s")
    
    # AR params
    t0 = time.perf_counter()
    Ab, Q = arhmm.resample_ar_params(
        seed, **data, **states, **params, **hypparams["ar_hypparams"]
    )
    sync(Ab)
    t_ar = time.perf_counter() - t0
    print(f"2. resample_ar_params:        {t_ar:6.2f} s  (func={arhmm.resample_ar_params.__name__})")
    
    # Discrete states
    t0 = time.perf_counter()
    z = arhmm.resample_discrete_stateseqs(seed, **data, **states, **params)
    sync(z)
    t_z = time.perf_counter() - t0
    print(f"3. resample_discrete_states:  {t_z:6.2f} s  (func={arhmm.resample_discrete_stateseqs.__name__})")
    
    # Full resample_model
    t0 = time.perf_counter()
    model = keypoint_slds.resample_model(data, **model, ar_only=True)
    sync(model)
    t_full = time.perf_counter() - t0
    print(f"4. keypoint_slds.resample_model(ar_only=True): {t_full:6.2f} s")
    
    # Check for NaNs
    t0 = time.perf_counter()
    from keypoint_moseq.fitting import check_for_nans
    any_nan, _, _ = check_for_nans(model)
    t_nan = time.perf_counter() - t0
    print(f"5. check_for_nans:            {t_nan*1000:6.2f} ms")
    
    print("\n=======================================================")

if __name__ == "__main__":
    main()
