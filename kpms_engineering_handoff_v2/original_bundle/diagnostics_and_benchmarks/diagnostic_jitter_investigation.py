import os
import sys
import inspect
import time
import h5py
import numpy as np

# Use GPU with strict bounded memory fraction (max ~6 GB, preallocation disabled)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
from utils.resample_batching import pure_numba_resample_continuous_stateseqs

def run_jitter_diagnostic():
    print("=" * 80)
    print("  DIAGNOSTIC: JITTER IMPACT ON CONTINUOUS LATENT SAMPLER VARIANCE")
    print("=" * 80)

    # Load frozen slice (N=2, T=400) from stable checkpoint amph_prod_k150k_run_01
    ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_prod_k150k_run_01", "checkpoint.h5")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_benchmark_test_timing", "checkpoint.h5")

    N_SEQS = 2
    T_LEN = 400

    with h5py.File(ckpt_path, "r") as f:
        if "model_snapshots" in f:
            snaps = sorted([int(k) for k in f["model_snapshots"].keys()])
            snap_grp = f["model_snapshots"][str(snaps[-1])]
        else:
            snap_grp = f["model"]

        Cd_np = np.array(snap_grp["params"]["Cd"][:], dtype=np.float64)
        sigmasq_np = np.array(snap_grp["params"]["sigmasq"][:], dtype=np.float64)
        Ab_np = np.array(snap_grp["params"]["Ab"][:], dtype=np.float64)
        Q_np = np.array(snap_grp["params"]["Q"][:], dtype=np.float64)

        v_np = np.array(snap_grp["states"]["v"][:N_SEQS, :T_LEN], dtype=np.float64)
        h_np = np.array(snap_grp["states"]["h"][:N_SEQS, :T_LEN], dtype=np.float64)
        s_np = np.array(snap_grp["states"]["s"][:N_SEQS, :T_LEN], dtype=np.float64)
        nlags = 3
        z_np = np.array(snap_grp["states"]["z"][:N_SEQS, :T_LEN - nlags], dtype=np.int32)

        Y_np = np.array(f["data"]["Y"][:N_SEQS, :T_LEN], dtype=np.float64)
        mask_np = np.array(f["data"]["mask"][:N_SEQS, :T_LEN]).astype(bool)

    Cd = jnp.array(Cd_np)
    sigmasq = jnp.array(sigmasq_np)
    Ab = jnp.array(Ab_np)
    Q = jnp.array(Q_np)
    v = jnp.array(v_np)
    h = jnp.array(h_np)
    s = jnp.array(s_np)
    z = jnp.array(z_np)
    Y = jnp.array(Y_np)
    mask = jnp.array(mask_np)

    N, T, k, d = Y_np.shape
    D = Ab_np.shape[1]

    kalman_fn = slds_gibbs.resample_continuous_stateseqs
    numba_fn = pure_numba_resample_continuous_stateseqs

    N_DRAWS = 50
    print(f"Slice: N={N}, T={T}, Keypoints={k}, D={D}, Valid frames={int(np.sum(mask_np))}/{N*T}")
    print(f"Sampling budget: 4 conditions x {N_DRAWS} draws = 200 draws total.")

    # 1. Condition 1: Upstream Kalman with jitter = 1e-3
    print("\n[Condition 1/4] Upstream Kalman (jitter = 1e-3)...", flush=True)
    kalman_j1e3 = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(1000 + i)
        x = kalman_fn(
            seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
            jitter=1e-3, parallel_message_passing=False
        )
        kalman_j1e3[i] = np.array(x)
    print(f"  Done in {time.time() - t0:.2f}s.")

    # 2. Condition 2: Numba with jitter = 1e-3
    print("\n[Condition 2/4] Numba Banded Cholesky (jitter = 1e-3)...", flush=True)
    numba_j1e3 = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(2000 + i)
        x = numba_fn(
            seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
            jitter=1e-3
        )
        numba_j1e3[i] = np.array(x)
    print(f"  Done in {time.time() - t0:.2f}s.")

    # 3. Condition 3: Upstream Kalman with jitter = 0.0
    print("\n[Condition 3/4] Upstream Kalman (jitter = 0.0)...", flush=True)
    kalman_j0 = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    kalman_j0_failed = False
    kalman_j0_err = None
    try:
        for i in range(N_DRAWS):
            seed = jr.PRNGKey(3000 + i)
            x = kalman_fn(
                seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
                jitter=0.0, parallel_message_passing=False
            )
            kalman_j0[i] = np.array(x)
        print(f"  Done in {time.time() - t0:.2f}s.")
    except Exception as e:
        kalman_j0_failed = True
        kalman_j0_err = str(e)
        print(f"  FAILED with exception: {e}")

    # 4. Condition 4: Numba with jitter = 0.0
    print("\n[Condition 4/4] Numba Banded Cholesky (jitter = 0.0)...", flush=True)
    numba_j0 = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    numba_j0_failed = False
    numba_j0_err = None
    repairs_j0 = 0
    try:
        for i in range(N_DRAWS):
            seed = jr.PRNGKey(4000 + i)
            x, rep = numba_fn(
                seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
                jitter=0.0, return_diagnostics=True
            )
            numba_j0[i] = np.array(x)
            repairs_j0 += rep
        print(f"  Done in {time.time() - t0:.2f}s (Pivot repairs: {repairs_j0}).")
    except Exception as e:
        numba_j0_failed = True
        numba_j0_err = str(e)
        print(f"  FAILED with exception: {e}")

    # Check for NaNs
    nan_k_1e3 = bool(np.isnan(kalman_j1e3).any())
    nan_n_1e3 = bool(np.isnan(numba_j1e3).any())
    nan_k_0 = bool(np.isnan(kalman_j0).any()) if not kalman_j0_failed else True
    nan_n_0 = bool(np.isnan(numba_j0).any()) if not numba_j0_failed else True

    print("\nNaN checks:")
    print(f"  Kalman (jitter=1e-3): {nan_k_1e3}")
    print(f"  Numba  (jitter=1e-3): {nan_n_1e3}")
    print(f"  Kalman (jitter=0.0):  {nan_k_0} (Failed: {kalman_j0_failed})")
    print(f"  Numba  (jitter=0.0):  {nan_n_0} (Failed: {numba_j0_failed})")

    # Compute Variances (ddof=1) across draws
    var_k_1e3 = np.var(kalman_j1e3, axis=0, ddof=1)
    var_n_1e3 = np.var(numba_j1e3, axis=0, ddof=1)
    var_k_0 = np.var(kalman_j0, axis=0, ddof=1) if not kalman_j0_failed else None
    var_n_0 = np.var(numba_j0, axis=0, ddof=1) if not numba_j0_failed else None

    # Overall pooled means on valid frames
    mvar_k_1e3 = float(np.mean(var_k_1e3[mask_np]))
    mvar_n_1e3 = float(np.mean(var_n_1e3[mask_np]))
    ratio_1e3 = mvar_n_1e3 / mvar_k_1e3

    mvar_k_0 = float(np.mean(var_k_0[mask_np])) if var_k_0 is not None else float("nan")
    mvar_n_0 = float(np.mean(var_n_0[mask_np])) if var_n_0 is not None else float("nan")
    ratio_0 = mvar_n_0 / mvar_k_0 if (var_k_0 is not None and var_n_0 is not None) else float("nan")

    # Per-dimension variances
    dim_k_1e3 = np.zeros(D)
    dim_n_1e3 = np.zeros(D)
    dim_k_0 = np.zeros(D)
    dim_n_0 = np.zeros(D)
    dim_ratio_1e3 = np.zeros(D)
    dim_ratio_0 = np.zeros(D)

    for d_idx in range(D):
        dim_k_1e3[d_idx] = float(np.mean(var_k_1e3[:, :, d_idx][mask_np]))
        dim_n_1e3[d_idx] = float(np.mean(var_n_1e3[:, :, d_idx][mask_np]))
        dim_ratio_1e3[d_idx] = dim_n_1e3[d_idx] / dim_k_1e3[d_idx]

        if var_k_0 is not None and var_n_0 is not None:
            dim_k_0[d_idx] = float(np.mean(var_k_0[:, :, d_idx][mask_np]))
            dim_n_0[d_idx] = float(np.mean(var_n_0[:, :, d_idx][mask_np]))
            dim_ratio_0[d_idx] = dim_n_0[d_idx] / dim_k_0[d_idx]

    print("\n" + "=" * 90)
    print(f"{'JITTER SENSITIVITY DIAGNOSTIC RESULTS':^90}")
    print("=" * 90)
    print(f"\n[A] Overall Pooled Variances & Ratios (Valid Frames):")
    print(f"  * Jitter = 1e-3:")
    print(f"      Kalman Mean Var: {mvar_k_1e3:.6f}")
    print(f"      Numba Mean Var:  {mvar_n_1e3:.6f}")
    print(f"      Ratio (Numba / Kalman): {ratio_1e3:.4f}")
    print(f"  * Jitter = 0.0:")
    print(f"      Kalman Mean Var: {mvar_k_0:.6f}")
    print(f"      Numba Mean Var:  {mvar_n_0:.6f}")
    print(f"      Ratio (Numba / Kalman): {ratio_0:.4f}")
    print(f"  * Ratio Change (Jitter 1e-3 -> 0.0): {ratio_0 - ratio_1e3:+.4f}")

    print(f"\n[B] Per-Dimension Comparison Table:")
    header = f"{'Dim':<5} | {'Kalman (1e-3)':<14} | {'Numba (1e-3)':<14} | {'Ratio (1e-3)':<12} | {'Kalman (0.0)':<14} | {'Numba (0.0)':<14} | {'Ratio (0.0)':<12}"
    print(header)
    print("-" * 90)
    for d_idx in range(D):
        row = f"D{d_idx:<4} | {dim_k_1e3[d_idx]:<14.5f} | {dim_n_1e3[d_idx]:<14.5f} | {dim_ratio_1e3[d_idx]:<12.4f} | {dim_k_0[d_idx]:<14.5f} | {dim_n_0[d_idx]:<14.5f} | {dim_ratio_0[d_idx]:<12.4f}"
        print(row)
    print("=" * 90)

if __name__ == "__main__":
    run_jitter_diagnostic()
