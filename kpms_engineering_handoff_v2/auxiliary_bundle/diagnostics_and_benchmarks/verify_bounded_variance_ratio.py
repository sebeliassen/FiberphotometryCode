import inspect
import os
import sys
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

import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
from utils.resample_batching import pure_numba_resample_continuous_stateseqs


def run_bounded_experiment():
    print("=" * 80)
    print("  BOUNDED CONTINUOUS LATENT SAMPLER VARIANCE RATIO AUDIT (300 DRAWS)")
    print("=" * 80)

    # 1. Verify Callables and Source Locations
    kalman_fn = slds_gibbs.resample_continuous_stateseqs
    numba_fn = pure_numba_resample_continuous_stateseqs

    kalman_py_fn = getattr(kalman_fn, 'python_function', getattr(kalman_fn, '_fun', kalman_fn))
    kalman_file = inspect.getfile(kalman_py_fn)
    numba_file = inspect.getfile(numba_fn)

    print("\n[Step 1] Callable Verification & Source Locations:")
    print(f"  * Reference Callable (Kalman): {kalman_fn.__name__}")
    print(f"    Source: {kalman_file}")
    print(f"  * Comparison Callable (Numba): {numba_fn.__name__}")
    print(f"    Source: {numba_file}")

    assert "jax_moseq" in kalman_file, f"Reference callable is not upstream jax_moseq! Got: {kalman_file}"
    assert "resample_batching.py" in numba_file, f"Comparison callable is not Numba sampler! Got: {numba_file}"

    # 2. Load Checkpoint and Freeze Small Slice (N=2, T=400)
    # Using completed, stable checkpoint amph_prod_k150k_run_01
    ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_prod_k150k_run_01", "checkpoint.h5")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_benchmark_test_timing", "checkpoint.h5")

    print(f"\n[Step 2] Loading frozen slice from checkpoint: {ckpt_path}")
    N_SEQS = 2
    T_LEN = 400

    with h5py.File(ckpt_path, "r") as f:
        # Load snapshot 150 or latest available snapshot
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

    # Convert to JAX arrays for Kalman reference
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
    valid_count = int(np.sum(mask_np))
    total_count = N * T
    print(f"  * Frozen Slice: N={N}, T={T}, Keypoints={k}, Latent Dim D={D}, AR lags={nlags}")
    print(f"  * Production-valid frames: {valid_count} / {total_count} ({100.0 * valid_count / total_count:.2f}%)")
    print("  * Artificial gaps injected: NONE (evaluating production-valid frames only)")

    # 3. Generate 3 Independent Pools of 100 Draws (Total budget: 300 draws)
    N_DRAWS = 100
    jitter = 1e-3

    print(f"\n[Step 3] Sampling 3 independent pools of {N_DRAWS} draws each (Total: 300 draws)...")

    # Warmup / JIT compilation pass
    print("  * JIT warmup pass...", flush=True)
    _ = kalman_fn(
        jr.PRNGKey(0), Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
        jitter=jitter, parallel_message_passing=False
    ).block_until_ready()
    _ = pure_numba_resample_continuous_stateseqs(
        jr.PRNGKey(1), Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
        jitter=jitter
    )

    # Pool 1: Upstream Kalman A (Seeds: 10,000 .. 10,099)
    print("  * Generating Pool 1: Upstream Kalman A (100 draws)...", flush=True)
    kalman_A = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(10000 + i)
        x = kalman_fn(
            seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
            jitter=jitter, parallel_message_passing=False
        )
        kalman_A[i] = np.array(x)
    t_ka = time.time() - t0
    print(f"    Done in {t_ka:.2f}s ({t_ka/N_DRAWS*1000:.1f} ms/draw).")

    # Pool 2: Upstream Kalman B (Seeds: 20,000 .. 20,099) - Sampling-noise control
    print("  * Generating Pool 2: Upstream Kalman B (100 draws, Control)...", flush=True)
    kalman_B = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(20000 + i)
        x = kalman_fn(
            seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
            jitter=jitter, parallel_message_passing=False
        )
        kalman_B[i] = np.array(x)
    t_kb = time.time() - t0
    print(f"    Done in {t_kb:.2f}s ({t_kb/N_DRAWS*1000:.1f} ms/draw).")

    # Pool 3: Current Numba (Seeds: 30,000 .. 30,099)
    print("  * Generating Pool 3: Current Numba (100 draws)...", flush=True)
    numba_samples = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(30000 + i)
        x = pure_numba_resample_continuous_stateseqs(
            seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q,
            jitter=jitter
        )
        numba_samples[i] = np.array(x)
    t_n = time.time() - t0
    print(f"    Done in {t_n:.2f}s ({t_n/N_DRAWS*1000:.1f} ms/draw).")

    # 4. Evaluate Production-Valid Frames Only, ddof=1
    print("\n[Step 4] Computing point variances across draws (ddof=1) on valid frames...")
    var_A = np.var(kalman_A, axis=0, ddof=1)       # (N, T, D)
    var_B = np.var(kalman_B, axis=0, ddof=1)       # (N, T, D)
    var_N = np.var(numba_samples, axis=0, ddof=1)   # (N, T, D)

    # Average over valid coordinates
    mean_var_A = np.mean(var_A[mask_np])
    mean_var_B = np.mean(var_B[mask_np])
    mean_var_N = np.mean(var_N[mask_np])

    overall_ratio_N_A = mean_var_N / mean_var_A
    overall_ratio_B_A = mean_var_B / mean_var_A

    # Per-dimension variances and ratios
    dim_var_A = np.zeros(D)
    dim_var_B = np.zeros(D)
    dim_var_N = np.zeros(D)
    dim_ratio_N_A = np.zeros(D)
    dim_ratio_B_A = np.zeros(D)

    for d_idx in range(D):
        dim_var_A[d_idx] = np.mean(var_A[:, :, d_idx][mask_np])
        dim_var_B[d_idx] = np.mean(var_B[:, :, d_idx][mask_np])
        dim_var_N[d_idx] = np.mean(var_N[:, :, d_idx][mask_np])
        dim_ratio_N_A[d_idx] = dim_var_N[d_idx] / dim_var_A[d_idx]
        dim_ratio_B_A[d_idx] = dim_var_B[d_idx] / dim_var_A[d_idx]

    # 5. Bootstrap Estimation (500 Resamples of whole trajectories)
    print("\n[Step 5] Running 500 bootstrap resamples (resampling whole draw trajectories)...")
    N_BOOT = 500
    boot_rng = np.random.default_rng(12345)

    boot_ratio_N_A = np.zeros(N_BOOT)
    boot_ratio_B_A = np.zeros(N_BOOT)
    boot_dim_ratio_N_A = np.zeros((N_BOOT, D))
    boot_dim_ratio_B_A = np.zeros((N_BOOT, D))

    for b in range(N_BOOT):
        idx_A = boot_rng.choice(N_DRAWS, size=N_DRAWS, replace=True)
        idx_B = boot_rng.choice(N_DRAWS, size=N_DRAWS, replace=True)
        idx_N = boot_rng.choice(N_DRAWS, size=N_DRAWS, replace=True)

        b_var_A = np.var(kalman_A[idx_A], axis=0, ddof=1)
        b_var_B = np.var(kalman_B[idx_B], axis=0, ddof=1)
        b_var_N = np.var(numba_samples[idx_N], axis=0, ddof=1)

        b_mean_A = np.mean(b_var_A[mask_np])
        b_mean_B = np.mean(b_var_B[mask_np])
        b_mean_N = np.mean(b_var_N[mask_np])

        boot_ratio_N_A[b] = b_mean_N / b_mean_A
        boot_ratio_B_A[b] = b_mean_B / b_mean_A

        for d_idx in range(D):
            b_d_A = np.mean(b_var_A[:, :, d_idx][mask_np])
            b_d_B = np.mean(b_var_B[:, :, d_idx][mask_np])
            b_d_N = np.mean(b_var_N[:, :, d_idx][mask_np])
            boot_dim_ratio_N_A[b, d_idx] = b_d_N / b_d_A
            boot_dim_ratio_B_A[b, d_idx] = b_d_B / b_d_A

    ci_N_A = np.percentile(boot_ratio_N_A, [2.5, 97.5])
    ci_B_A = np.percentile(boot_ratio_B_A, [2.5, 97.5])

    ci_dim_N_A = np.percentile(boot_dim_ratio_N_A, [2.5, 97.5], axis=0) # (2, D)

    # 6. Report Results
    print("\n" + "=" * 80)
    print("  BOUNDED VARIANCE RATIO AUDIT RESULTS")
    print("=" * 80)
    print(f"\n[A] Overall Pooled Ratios across all {D} Latent Dimensions (Valid Frames):")
    print(f"  * Control (Kalman B / Kalman A): {overall_ratio_B_A:.4f}  [95% CI: {ci_B_A[0]:.4f} - {ci_B_A[1]:.4f}]")
    print(f"  * Comparison (Numba / Kalman A): {overall_ratio_N_A:.4f}  [95% CI: {ci_N_A[0]:.4f} - {ci_N_A[1]:.4f}]")
    includes_1_ctrl = (ci_B_A[0] <= 1.0 <= ci_B_A[1])
    includes_1_numba = (ci_N_A[0] <= 1.0 <= ci_N_A[1])
    print(f"  * Control CI includes 1.0: {includes_1_ctrl}")
    print(f"  * Numba CI includes 1.0:   {includes_1_numba}")

    print("\n[B] Per-Dimension Variance Ratios (Numba / Kalman A):")
    print(f"{'Dim':<5} | {'Var Kalman A':<14} | {'Var Numba':<14} | {'Ratio (N/A)':<12} | {'95% Bootstrap CI':<24}")
    print("-" * 75)
    for d_idx in range(D):
        ci_str = f"[{ci_dim_N_A[0, d_idx]:.4f}, {ci_dim_N_A[1, d_idx]:.4f}]"
        print(f"D{d_idx:<4} | {dim_var_A[d_idx]:<14.5f} | {dim_var_N[d_idx]:<14.5f} | {dim_ratio_N_A[d_idx]:<12.4f} | {ci_str:<24}")
    print("=" * 80)

if __name__ == "__main__":
    run_bounded_experiment()
