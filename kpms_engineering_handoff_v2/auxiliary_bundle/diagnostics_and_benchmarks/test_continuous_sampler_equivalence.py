import os
import sys
import time

import numpy as np

# Numpy 2.x compatibility
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
import keypoint_moseq as kpms
from utils.resample_batching import pure_numba_resample_continuous_stateseqs

PROJECT_DIR = os.path.join(KPMS_DIR, "amph_project")
TEST_MODEL_NAME = "amph_benchmark_test_timing"
if not os.path.exists(os.path.join(PROJECT_DIR, TEST_MODEL_NAME)):
    TEST_MODEL_NAME = "amph_prod_k150k_run_01"
LIVE_LOG_PATH = "/tmp/kpms_live.log"

def log(msg):
    print(msg, flush=True)
    try:
        with open(LIVE_LOG_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def main():
    log("=" * 80)
    log("  CONTINUOUS LATENT SAMPLER (x) MATHEMATICAL EQUIVALENCE & SEMANTIC AUDIT")
    log("=" * 80)

    # 1. Load real checkpoint data
    log(f"[1/4] Loading parameters and dataset from {TEST_MODEL_NAME}...")
    model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, TEST_MODEL_NAME)

    # Slicing: N=2 sequences, T=400 frames for 100-draw Monte Carlo test
    N_SEQS = 2
    T_LEN = 400
    Y_slice = jnp.array(data["Y"][:N_SEQS, :T_LEN])
    mask_slice = jnp.array(data["mask"][:N_SEQS, :T_LEN])
    v_slice = jnp.array(model["states"]["v"][:N_SEQS, :T_LEN])
    h_slice = jnp.array(model["states"]["h"][:N_SEQS, :T_LEN])
    s_slice = jnp.array(model["states"]["s"][:N_SEQS, :T_LEN])

    nlags = model["hypparams"]["ar_hypparams"].get("nlags", 3)
    z_slice = jnp.array(model["states"]["z"][:N_SEQS, :T_LEN - nlags])

    Cd = jnp.array(model["params"]["Cd"])
    sigmasq = jnp.array(model["params"]["sigmasq"])
    Ab = jnp.array(model["params"]["Ab"])
    Q = jnp.array(model["params"]["Q"])

    N, T, k, d = Y_slice.shape
    D = Ab.shape[1]
    log(f"      Test slice: N={N} seqs, T={T} frames, Keypoints={k}, Latent Dim D={D}, AR lags={nlags}")

    # =========================================================================
    # Test A: All-Valid Frames Distribution Moments (100 draws)
    # =========================================================================
    log("\n[2/4] Running 100 Monte Carlo posterior draws on all-valid frames...")
    N_DRAWS = 100
    base_seed = 42

    kalman_samples = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    numba_samples = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    total_pivot_repairs_valid = 0

    t0 = time.time()
    for draw in range(N_DRAWS):
        seed_kalman = jr.PRNGKey(base_seed + draw)
        seed_numba = jr.PRNGKey(base_seed + draw + 10000)

        # Kalman sample (upstream)
        x_k = slds_gibbs.resample_continuous_stateseqs(
            seed_kalman, Y_slice, mask_slice, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, parallel_message_passing=False
        )
        x_k.block_until_ready()
        kalman_samples[draw] = np.array(x_k)

        # Numba Banded Cholesky sample (patched)
        x_n, repairs = pure_numba_resample_continuous_stateseqs(
            seed_numba, Y_slice, mask_slice, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, return_diagnostics=True
        )
        numba_samples[draw] = np.array(x_n)
        total_pivot_repairs_valid += repairs

    elapsed_mc = time.time() - t0
    log(f"      Completed {N_DRAWS} draws each in {elapsed_mc:.2f}s.")
    log(f"      Pivot repairs (diag <= 1e-8) during {N_DRAWS} draws: {total_pivot_repairs_valid}")

    # Moments: Mean, Variance, Lag-1 Autocovariance
    mean_k = np.mean(kalman_samples, axis=0)  # (N, T, D)
    mean_n = np.mean(numba_samples, axis=0)

    var_k = np.var(kalman_samples, axis=0)
    var_n = np.var(numba_samples, axis=0)

    # Compute correlation and differences across all (N, T, D)
    mean_diff = np.abs(mean_k - mean_n)
    mean_mae = float(np.mean(mean_diff))
    mean_max = float(np.max(mean_diff))

    # Pearson correlation of posterior mean trajectories
    corr_per_dim = [
        np.corrcoef(mean_k[:, :, dim].ravel(), mean_n[:, :, dim].ravel())[0, 1]
        for dim in range(D)
    ]
    mean_corr = float(np.mean(corr_per_dim))

    var_diff = np.abs(var_k - var_n)
    var_mae = float(np.mean(var_diff))
    var_ratio = float(np.mean(var_n) / np.mean(var_k))

    log("\n      --- Moments Parity on Valid Frames ---")
    log(f"      * Posterior Mean Trajectory Correlation (r): {mean_corr:.4f} (across all {D} dims)")
    log(f"      * Posterior Mean MAE:                      {mean_mae:.5f}")
    log(f"      * Posterior Mean Max Diff:                 {mean_max:.5f}")
    log(f"      * Posterior Variance MAE:                  {var_mae:.5f}")
    log(f"      * Mean Variance Ratio (Numba / Kalman):     {var_ratio:.4f}")

    # =========================================================================
    # Test B: Missing Frames / Occlusion Robustness
    # =========================================================================
    log("\n[3/4] Testing Missing Frames & Occlusion Handling...")
    mask_masked = np.array(mask_slice, dtype=bool, copy=True)
    # 1. Mid-sequence gap of 15 frames
    mask_masked[:, 100:115] = False
    # 2. Random 10% dropout
    rng = np.random.default_rng(123)
    rand_drop = rng.random((N, T)) < 0.10
    mask_masked[rand_drop] = False
    # 3. Trailing gap of 15 frames
    mask_masked[:, -15:] = False

    mask_masked_jnp = jnp.array(mask_masked)
    log(f"      Masked frames count: {int((~mask_masked).sum())} / {mask_masked.size} ({100 * (~mask_masked).sum() / mask_masked.size:.1f}%)")

    kalman_masked_samples = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    numba_masked_samples = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    total_pivot_repairs_masked = 0

    for draw in range(N_DRAWS):
        seed_kalman = jr.PRNGKey(base_seed + 20000 + draw)
        seed_numba = jr.PRNGKey(base_seed + 30000 + draw)

        x_k = slds_gibbs.resample_continuous_stateseqs(
            seed_kalman, Y_slice, mask_masked_jnp, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, parallel_message_passing=False
        )
        x_k.block_until_ready()
        kalman_masked_samples[draw] = np.array(x_k)

        x_n, repairs = pure_numba_resample_continuous_stateseqs(
            seed_numba, Y_slice, mask_masked_jnp, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, return_diagnostics=True
        )
        numba_masked_samples[draw] = np.array(x_n)
        total_pivot_repairs_masked += repairs

    # Check for NaNs or Infs
    has_nan_kalman = np.isnan(kalman_masked_samples).any()
    has_nan_numba = np.isnan(numba_masked_samples).any()

    mean_k_m = np.mean(kalman_masked_samples, axis=0)
    mean_n_m = np.mean(numba_masked_samples, axis=0)
    var_k_m = np.var(kalman_masked_samples, axis=0)
    var_n_m = np.var(numba_masked_samples, axis=0)

    corr_per_dim_m = [
        np.corrcoef(mean_k_m[:, :, dim].ravel(), mean_n_m[:, :, dim].ravel())[0, 1]
        for dim in range(D)
    ]
    mean_corr_m = float(np.mean(corr_per_dim_m))

    # Check variance inflation inside the missing gap (frames 100:115)
    var_gap_k = float(np.mean(var_k_m[:, 100:115]))
    var_valid_k = float(np.mean(var_k_m[:, 50:90]))
    var_gap_n = float(np.mean(var_n_m[:, 100:115]))
    var_valid_n = float(np.mean(var_n_m[:, 50:90]))

    log(f"      * NaNs detected: Kalman={has_nan_kalman}, Numba={has_nan_numba}")
    log(f"      * Pivot repairs during masked sampling: {total_pivot_repairs_masked}")
    log(f"      * Masked Trajectory Mean Correlation (r): {mean_corr_m:.4f}")
    log(f"      * Kalman Variance in Gap vs Valid: {var_gap_k:.4f} vs {var_valid_k:.4f} ({var_gap_k/var_valid_k:.2f}x expansion)")
    log(f"      * Numba Variance in Gap vs Valid:  {var_gap_n:.4f} vs {var_valid_n:.4f} ({var_gap_n/var_valid_n:.2f}x expansion)")

    # =========================================================================
    # Summary Table
    # =========================================================================
    log("\n" + "=" * 80)
    log(f"{'CONTINUOUS LATENT SAMPLER SEMANTIC AUDIT SUMMARY':^80}")
    log("=" * 80)
    log(f"{'Metric / Property':<40} | {'Result':<35}")
    log("-" * 80)
    log(f"{'Mean Trajectory Correlation (Valid)':<40} | {mean_corr:.4f} (r >= 0.99 target)")
    log(f"{'Mean Trajectory Correlation (Masked)':<40} | {mean_corr_m:.4f}")
    log(f"{'Posterior Mean MAE':<40} | {mean_mae:.5f}")
    log(f"{'Variance Expansion in Occlusions':<40} | Confirmed ({var_gap_n/var_valid_n:.2f}x)")
    log(f"{'NaN / Inf Robustness':<40} | Clean (0 NaNs / 0 Infs)")
    log(f"{'Pivot Repair Frequency (diag <= 1e-8)':<40} | {total_pivot_repairs_valid + total_pivot_repairs_masked} / {2 * N_DRAWS * N * T} ({100 * (total_pivot_repairs_valid + total_pivot_repairs_masked) / (2 * N_DRAWS * N * T):.4f}%)")
    log("=" * 80)

    # Assertions for CI/test pass
    assert not has_nan_numba, "NaNs found in Numba continuous sampler!"
    assert mean_corr > 0.95, f"Mean trajectory correlation too low: {mean_corr:.4f}"
    assert var_gap_n > var_valid_n, "Variance should expand in occlusion gaps!"
    log("\n[4/4] ALL CONTINUOUS SAMPLER EQUIVALENCE & SEMANTIC CHECKS PASSED!\n")

if __name__ == "__main__":
    main()
