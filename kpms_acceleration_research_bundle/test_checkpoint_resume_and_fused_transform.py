#!/usr/bin/env python3
"""
diagnostics/test_checkpoint_resume_and_fused_transform.py

Tests Point 4 from reviewer feedback:
1. Fused transform parity with a reused output buffer and a partial chunk (N=13).
2. Checkpoint resume continuity:
   - Compares uninterrupted execution (Iteration 1 -> 2)
   - Against reload from HDF5 checkpoint, cache rebuild, and resumed execution (Iter 1 saved -> Reload -> Iter 2).
   - Verifies numerical agreement under identical random seed.
"""

import sys, os, time, shutil, tempfile
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode')
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll')

import numpy as np
if not hasattr(np, "bool8"): np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import jax.random as jr
import keypoint_moseq as kpms
from jax_moseq.models import keypoint_slds
import utils.resample_batching as rb
from utils.resample_batching import KPMSStateCache, _numba_fused_rigid_transform

def numpy_transform_reference(Y_c, v_c, h_c):
    N, T = Y_c.shape[:2]
    cos_h = np.cos(h_c)[:, :, None]
    sin_h = np.sin(h_c)[:, :, None]
    Y_trans = Y_c - v_c[:, :, None, :]
    Y_rot_0 = cos_h * Y_trans[..., 0] + sin_h * Y_trans[..., 1]
    Y_rot_1 = -sin_h * Y_trans[..., 0] + cos_h * Y_trans[..., 1]
    return np.stack([Y_rot_0, Y_rot_1], axis=-1).reshape(N, T, -1)

def test_fused_transform_reused_and_partial():
    print(">>> 1. AUDITING FUSED RIGID TRANSFORM (REUSED BUFFER & PARTIAL CHUNK) <<<")
    rng = np.random.default_rng(101)
    
    # Test cases: Full chunk (N=50) and Partial chunk (N=13)
    for N_chunk in [50, 13]:
        T = 1000
        K = 17
        d = 2

        Y_c = rng.standard_normal((N_chunk, T, K, d))
        v_c = rng.standard_normal((N_chunk, T, d))
        h_c = rng.standard_normal((N_chunk, T))

        # Reused buffer preallocated to max_chunk=50
        out_buffer = np.zeros((50, T, K * d), dtype=np.float64)

        # Run multiple passes into the same buffer to test reuse & dirty-state resilience
        for pass_idx in range(3):
            out_view = out_buffer[:N_chunk]
            out_view.fill(-999.0)  # Dirty state
            _numba_fused_rigid_transform(Y_c, v_c, h_c, out_view)

            ref_van = numpy_transform_reference(Y_c, v_c, h_c)
            diff = np.max(np.abs(out_view - ref_van))
            assert diff < 1e-13, f"Pass {pass_idx} for N={N_chunk} failed! diff={diff}"

        print(f"  * N={N_chunk:2d} (partial/reused pass x3): Max diff = {diff:.3e} (Passed!)")
    print("--> Fused Rigid Transform verified across reused buffers and partial chunks.\n")

def test_checkpoint_resume_continuity():
    print(">>> 2. AUDITING CHECKPOINT RESUME CONTINUITY <<<")
    rb.apply_optimizations()

    # Load small slice from production checkpoint
    PROJECT_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
    SOURCE_MODEL = "amph_benchmark_test_timing"
    if not os.path.exists(os.path.join(PROJECT_DIR, SOURCE_MODEL)):
        SOURCE_MODEL = "amph_prod_k150k_run_01"
    base_model, base_data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, SOURCE_MODEL)

    N_test, T_test = 4, 300
    test_data = {
        "Y": np.asarray(base_data["Y"][:N_test, :T_test]),
        "mask": np.asarray(base_data["mask"][:N_test, :T_test], dtype=bool),
        "conf": np.asarray(base_data["conf"][:N_test, :T_test])
    }

    test_model = {
        "seed": jr.PRNGKey(42),
        "states": {
            "x": jnp.asarray(base_model["states"]["x"][:N_test, :T_test]),
            "z": jnp.asarray(base_model["states"]["z"][:N_test, :T_test - 3]),
            "v": jnp.asarray(base_model["states"]["v"][:N_test, :T_test]),
            "h": jnp.asarray(base_model["states"]["h"][:N_test, :T_test]),
            "s": jnp.asarray(base_model["states"]["s"][:N_test, :T_test]),
        },
        "params": {
            "Ab": jnp.asarray(base_model["params"]["Ab"]),
            "Q": jnp.asarray(base_model["params"]["Q"]),
            "pi": jnp.asarray(base_model["params"]["pi"]),
            "betas": jnp.asarray(base_model["params"]["betas"]),
            "Cd": jnp.asarray(base_model["params"]["Cd"]),
            "sigmasq": jnp.asarray(base_model["params"]["sigmasq"]),
        },
        "hypparams": base_model["hypparams"],
        "noise_prior": base_model["noise_prior"][:N_test, :T_test] if hasattr(base_model["noise_prior"], "shape") and len(base_model["noise_prior"].shape) >= 2 else base_model["noise_prior"],
    }

    # Branch 1: Uninterrupted run (Iter 1 -> Iter 2)
    print("Running Branch 1: Uninterrupted 2 iterations...", flush=True)
    KPMSStateCache.clear()
    m1 = keypoint_slds.resample_model(test_data, **test_model, jitter=1e-3)
    m2_uninterrupted = keypoint_slds.resample_model(test_data, **m1, jitter=1e-3)

    # Branch 2: Iter 1 -> Save to HDF5 -> Wipe Caches -> Reload -> Iter 2
    print("Running Branch 2: Iter 1 -> Save HDF5 -> Cache Clear -> Reload -> Iter 2...", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_resume.h5")
        KPMSStateCache.clear()
        
        # Iteration 1
        m1_branch2 = keypoint_slds.resample_model(test_data, **test_model, jitter=1e-3)
        
        # Save snapshot 1 to HDF5
        kpms.save_hdf5(ckpt_path, m1_branch2, "model_snapshots/1", exist_ok=True)
        kpms.save_hdf5(ckpt_path, test_data, "data", exist_ok=True)
        kpms.save_hdf5(ckpt_path, (metadata[0][:N_test], metadata[1][:N_test]), "metadata", exist_ok=True)

        # Completely wipe all memory and runtime caches
        KPMSStateCache.clear()
        del m1_branch2

        # Reload from HDF5
        loaded_m1, loaded_data, _, loaded_iter = kpms.load_checkpoint(path=ckpt_path, iteration=1)
        assert loaded_iter == 1, f"Expected loaded iteration 1, got {loaded_iter}"

        # Rebuild runtime cache from loaded checkpoint
        loaded_data["Y"] = np.asarray(loaded_data["Y"])
        KPMSStateCache.clear()

        # Run resumed Iteration 2
        m2_resumed = keypoint_slds.resample_model(loaded_data, **loaded_m1, jitter=1e-3)

    # Compare Branch 1 and Branch 2 at Iteration 2
    print("\nComparing Uninterrupted vs Resumed Iteration 2 states:")
    
    # 1. Compare AR parameters (Ab, Q)
    diff_Ab = float(jnp.max(jnp.abs(m2_uninterrupted["params"]["Ab"] - m2_resumed["params"]["Ab"])))
    diff_Q = float(jnp.max(jnp.abs(m2_uninterrupted["params"]["Q"] - m2_resumed["params"]["Q"])))
    print(f"  * Ab max diff: {diff_Ab:.3e}")
    print(f"  * Q  max diff: {diff_Q:.3e}")
    assert diff_Ab < 1e-12, f"Ab parameter mismatch! {diff_Ab}"
    assert diff_Q < 1e-12, f"Q parameter mismatch! {diff_Q}"

    # 2. Compare continuous states (x)
    diff_x = float(jnp.max(jnp.abs(m2_uninterrupted["states"]["x"] - m2_resumed["states"]["x"])))
    print(f"  * x  max diff: {diff_x:.3e}")
    assert diff_x < 1e-12, f"Continuous states x mismatch! {diff_x}"

    # 3. Compare discrete states (z)
    mismatches_z = int(jnp.sum(m2_uninterrupted["states"]["z"] != m2_resumed["states"]["z"]))
    print(f"  * z mismatches: {mismatches_z} / {m2_uninterrupted['states']['z'].size}")
    assert mismatches_z == 0, f"Discrete states z mismatch! {mismatches_z}"

    print("--> Checkpoint Resume Continuity verified: Resumed execution matches uninterrupted run exactly!\n")

def main():
    print("=" * 80)
    print("  TEST 4: CHECKPOINT RESUME CONTINUITY & FUSED TRANSFORM AUDIT")
    print("=" * 80 + "\n", flush=True)

    test_fused_transform_reused_and_partial()
    test_checkpoint_resume_continuity()

    print("=" * 80)
    print("  ALL RESUME CONTINUITY AND FUSED TRANSFORM AUDITS PASSED!")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    main()
