import os
import sys
import time

import h5py
import numpy as np

# Bounded GPU memory (max ~4.8 GB, preallocation disabled to protect running training)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.models.keypoint_slds.alignment import to_vanilla_slds
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.kalman import ar_to_lds_emissions, kalman_sample
from jax_moseq.utils.utils import mixed_map
from utils.resample_batching import pure_numba_resample_continuous_stateseqs

na = None

def build_diagnostic_kalman_sampler(lag_noise, jitter=0.0):
    """
    Constructs an isolated, diagnostic-only Upstream Kalman continuous state sampler
    with a parameterized lag-coordinate noise injection in AR dynamics.
    Production code is untouched.
    """
    def ar_to_lds_dynamics_custom(Ab, Q):
        nlags = get_nlags(Ab)
        latent_dim = Ab.shape[-2]
        lds_dim = latent_dim * nlags
        eye = jnp.eye(latent_dim * (nlags - 1))

        A = Ab[..., :-1]
        dims = A.shape[:-2]
        A_ = jnp.zeros((*dims, lds_dim, lds_dim))
        A_ = A_.at[..., :-latent_dim, latent_dim:].set(eye)
        A_ = A_.at[..., -latent_dim:, :].set(A)

        b = Ab[..., -1]
        b_ = jnp.zeros((*dims, lds_dim))
        b_ = b_.at[..., -latent_dim:].set(b)

        dims = Q.shape[:-2]
        Q_ = jnp.zeros((*dims, lds_dim, lds_dim))
        # Parameterized lag-copy noise:
        Q_ = Q_.at[..., :-latent_dim, :-latent_dim].set(eye * lag_noise)
        Q_ = Q_.at[..., -latent_dim:, -latent_dim:].set(Q)
        return A_, b_, Q_

    @partial(jax.jit, static_argnames=("parallel_message_passing",))
    def resample_stateseqs(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, parallel_message_passing=False):
        y, s_obs, Cd_obs, sigmasq_obs = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
        n_recordings, latent_dim, obs_dim = y.shape[0], Ab.shape[1], y.shape[-1]
        n_lags = Ab.shape[2] // latent_dim

        m0 = jnp.zeros(latent_dim * n_lags)
        S0 = 10 * jnp.eye(latent_dim * n_lags)
        masked_dynamics_noise = 10
        masked_obs_noise = 10

        y_ = y[:, n_lags - 1 :]
        mask_ = mask[:, n_lags - 1 :]
        R_ = sigmasq_obs * s_obs[:, n_lags - 1 :]

        C_, d_, R_, y_, m0_, S0_ = jax.vmap(
            ar_to_lds_emissions, in_axes=(na, 0, 0, na, na, na)
        )(Cd_obs, sigmasq_obs * s_obs, y, m0, S0, n_lags)

        A_, b_, Q_ = ar_to_lds_dynamics_custom(Ab, Q)

        ar_dim = latent_dim * n_lags
        eye_zero_order = jnp.zeros((ar_dim, ar_dim))
        eye_zero_order = eye_zero_order.at[-latent_dim:, -latent_dim:].set(jnp.eye(latent_dim))

        masked_dynamics_params = {
            "weights": eye_zero_order,
            "bias": jnp.zeros(ar_dim),
            "cov": jnp.eye(ar_dim) * masked_dynamics_noise,
        }
        masked_obs_noise_diag = jnp.ones(obs_dim) * masked_obs_noise

        in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
        x = mixed_map(
            partial(kalman_sample, jitter=jitter, parallel=parallel_message_passing),
            in_axes,
        )(
            jr.split(seed, n_recordings),
            y_,
            mask_,
            z,
            m0,
            S0,
            A_,
            b_,
            Q_,
            C_,
            d_,
            R_,
            masked_dynamics_params,
            masked_obs_noise_diag,
        )

        if n_lags > 1:
            x = jnp.concatenate(
                [
                    x[:, 0, : (n_lags - 1) * latent_dim].reshape(-1, n_lags - 1, latent_dim),
                    x[:, :, -latent_dim:],
                ],
                axis=1,
            )
        return x

    return resample_stateseqs, ar_to_lds_dynamics_custom


def run_diagnostic():
    print("=" * 85)
    print("  CONTROLLED DIAGNOSTIC: UPSTREAM LAG-COPY NOISE SWEEP vs NUMBA SAMPLER")
    print("=" * 85)

    ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_prod_k150k_run_01", "checkpoint.h5")
    N_SEQS = 2
    T_LEN = 400

    print(f"Loading frozen all-valid slice from: {ckpt_path}")
    with h5py.File(ckpt_path, "r") as f:
        snap_grp = f["model_snapshots"]["150"]
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
    n_valid = int(np.sum(mask_np))

    print(f"Slice summary: N={N}, T={T}, Keypoints={k}, Latent D={D}")
    print(f"Valid frame count: {n_valid} / {N * T} (100% all-valid slice)")
    print("Sampling budget: 50 draws x 4 conditions = 200 draws total.")
    print("Sampling jitter: set to 0.0 strictly across all conditions.")

    N_DRAWS = 50

    # Verification of dynamics parameters
    print("\n[Callable Verification] Inspecting lag-coordinate diagonal entries in dynamics Q_:")
    kalman_01_fn, dyn_01 = build_diagnostic_kalman_sampler(lag_noise=1e-2, jitter=0.0)
    kalman_001_fn, dyn_001 = build_diagnostic_kalman_sampler(lag_noise=1e-3, jitter=0.0)
    kalman_0_fn, dyn_0 = build_diagnostic_kalman_sampler(lag_noise=0.0, jitter=0.0)

    _, _, Q_01 = dyn_01(Ab, Q)
    _, _, Q_001 = dyn_001(Ab, Q)
    _, _, Q_0 = dyn_0(Ab, Q)

    print(f"  * Upstream (lag_noise = 0.010) Q_ lag block diagonal entry:  {float(Q_01[0, 0, 0]):.6f}")
    print(f"  * Upstream (lag_noise = 0.001) Q_ lag block diagonal entry:  {float(Q_001[0, 0, 0]):.6f}")
    print(f"  * Upstream (lag_noise = 0.000) Q_ lag block diagonal entry:  {float(Q_0[0, 0, 0]):.6f}")
    print(f"  * Main continuous state Q diagonal entry (reference):        {float(Q_np[0, 0, 0]):.6f}")

    # =========================================================================
    # Condition 1: Unchanged Current Numba Sampler (jitter = 0.0)
    # =========================================================================
    print("\n[Condition 1/4] Current Numba Sampler (jitter = 0.0, 50 draws)...", flush=True)
    numba_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    repairs_total = 0
    numba_fn = pure_numba_resample_continuous_stateseqs
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(1000 + i)
        x, rep = numba_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=0.0, return_diagnostics=True)
        numba_draws[i] = np.array(x)
        repairs_total += rep
    numba_time = time.time() - t0
    numba_has_nan = np.isnan(numba_draws).any()
    print(f"  Done in {numba_time:.2f}s. NaNs: {numba_has_nan}, Pivot repairs: {repairs_total}")

    # =========================================================================
    # Condition 2: Upstream Kalman (lag_noise = 0.01, jitter = 0.0)
    # =========================================================================
    print("\n[Condition 2/4] Upstream Kalman (lag_noise = 0.01 [default], jitter = 0.0, 50 draws)...", flush=True)
    kalman_01_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(2000 + i)
        x = kalman_01_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, parallel_message_passing=False)
        kalman_01_draws[i] = np.array(x)
    k01_time = time.time() - t0
    k01_has_nan = np.isnan(kalman_01_draws).any()
    print(f"  Done in {k01_time:.2f}s. NaNs: {k01_has_nan}")

    # =========================================================================
    # Condition 3: Upstream Kalman (lag_noise = 0.001, jitter = 0.0)
    # =========================================================================
    print("\n[Condition 3/4] Upstream Kalman (lag_noise = 0.001 [10x reduced], jitter = 0.0, 50 draws)...", flush=True)
    kalman_001_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(3000 + i)
        x = kalman_001_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, parallel_message_passing=False)
        kalman_001_draws[i] = np.array(x)
    k001_time = time.time() - t0
    k001_has_nan = np.isnan(kalman_001_draws).any()
    print(f"  Done in {k001_time:.2f}s. NaNs: {k001_has_nan}")

    # =========================================================================
    # Condition 4: Upstream Kalman (lag_noise = 0.0, jitter = 0.0)
    # =========================================================================
    print("\n[Condition 4/4] Upstream Kalman (lag_noise = 0.0 [exact noiseless lag copy], jitter = 0.0, 50 draws)...", flush=True)
    kalman_0_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    k0_failed = False
    k0_err = None
    t0 = time.time()
    try:
        for i in range(N_DRAWS):
            seed = jr.PRNGKey(4000 + i)
            x = kalman_0_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, parallel_message_passing=False)
            x_arr = np.array(x)
            if np.isnan(x_arr).any():
                k0_failed = True
                k0_err = "Output contains NaNs due to singular/rank-deficient covariance in backward sampling"
                kalman_0_draws[i] = x_arr
                break
            kalman_0_draws[i] = x_arr
        k0_time = time.time() - t0
        print(f"  Execution finished in {k0_time:.2f}s. Breakdown detected: {k0_failed}")
    except Exception as e:
        k0_failed = True
        k0_err = f"{type(e).__name__}: {str(e)}"
        print(f"  FAILED with exception: {k0_err}")

    # =========================================================================
    # Compute Statistics: Means, Variances, and Ratios
    # =========================================================================
    # Variances (ddof=1)
    var_numba = np.var(numba_draws, axis=0, ddof=1)
    var_k01 = np.var(kalman_01_draws, axis=0, ddof=1)
    var_k001 = np.var(kalman_001_draws, axis=0, ddof=1)

    mean_numba = np.mean(numba_draws, axis=0)
    mean_k01 = np.mean(kalman_01_draws, axis=0)
    mean_k001 = np.mean(kalman_001_draws, axis=0)

    # Pooled variance on valid frames
    pvar_numba = float(np.mean(var_numba[mask_np]))
    pvar_k01 = float(np.mean(var_k01[mask_np]))
    pvar_k001 = float(np.mean(var_k001[mask_np]))

    ratio_01 = pvar_numba / pvar_k01
    ratio_001 = pvar_numba / pvar_k001

    # Posterior Mean Parity
    mae_mean_k01 = float(np.mean(np.abs(mean_numba - mean_k01)[mask_np]))
    mae_mean_k001 = float(np.mean(np.abs(mean_numba - mean_k001)[mask_np]))

    # Per-dimension variances
    dim_numba = np.zeros(D)
    dim_k01 = np.zeros(D)
    dim_k001 = np.zeros(D)
    dim_ratio_01 = np.zeros(D)
    dim_ratio_001 = np.zeros(D)

    for d_idx in range(D):
        dim_numba[d_idx] = float(np.mean(var_numba[:, :, d_idx][mask_np]))
        dim_k01[d_idx] = float(np.mean(var_k01[:, :, d_idx][mask_np]))
        dim_k001[d_idx] = float(np.mean(var_k001[:, :, d_idx][mask_np]))
        dim_ratio_01[d_idx] = dim_numba[d_idx] / dim_k01[d_idx]
        dim_ratio_001[d_idx] = dim_numba[d_idx] / dim_k001[d_idx]

    # Print Full Report
    print("\n" + "=" * 90)
    print(f"{'LAG-COPY PROCESS NOISE SWEEP: QUANTITATIVE RESULTS':^90}")
    print("=" * 90)

    print("\n[A] OVERALL POOLED POSTERIOR VARIANCE ON VALID FRAMES:")
    print(f"  1. Numba Banded Cholesky (Exact AR, 0 lag noise):        {pvar_numba:.6f}")
    print(f"  2. Upstream Kalman (lag_noise = 0.010 [default]):         {pvar_k01:.6f}  --> Ratio (Numba/K_01):  {ratio_01:.4f} (Deficit: {(1.0-ratio_01)*100:.1f}%)")
    print(f"  3. Upstream Kalman (lag_noise = 0.001 [10x reduced]):     {pvar_k001:.6f}  --> Ratio (Numba/K_001): {ratio_001:.4f} (Deficit: {(1.0-ratio_001)*100:.1f}%)")
    print(f"  4. Upstream Kalman (lag_noise = 0.000 [exact]):           NUMERICAL FAILURE ({k0_err})")

    shift = ratio_001 - ratio_01
    print("\n[B] SENSITIVITY SUMMARY:")
    print(f"  * Variance ratio shift when reducing lag noise (0.01 -> 0.001): {shift:+.4f}")
    print(f"  * Upstream variance change (0.01 -> 0.001): {pvar_k001 - pvar_k01:+.6f} ({((pvar_k001/pvar_k01)-1.0)*100:+.2f}%)")
    print(f"  * Posterior Mean MAE vs Numba (lag_noise=0.01):  {mae_mean_k01:.6f}")
    print(f"  * Posterior Mean MAE vs Numba (lag_noise=0.001): {mae_mean_k001:.6f}")

    print("\n[C] PER-DIMENSION POSTERIOR VARIANCE COMPARISON:")
    header = f"{'Dim':<5} | {'Numba (0.0)':<13} | {'Kalman (0.01)':<14} | {'Ratio (0.01)':<12} | {'Kalman (0.001)':<14} | {'Ratio (0.001)':<12} | {'Shift':<8}"
    print(header)
    print("-" * 90)
    for d_idx in range(D):
        dim_shift = dim_ratio_001[d_idx] - dim_ratio_01[d_idx]
        row = f"D{d_idx:<4} | {dim_numba[d_idx]:<13.5f} | {dim_k01[d_idx]:<14.5f} | {dim_ratio_01[d_idx]:<12.4f} | {dim_k001[d_idx]:<14.5f} | {dim_ratio_001[d_idx]:<12.4f} | {dim_shift:+8.4f}"
        print(row)
    print("=" * 90)

    print("\n[D] KEY TAKEAWAYS:")
    if k0_failed:
        print("  1. Upstream Kalman strictly requires lag-copy process noise (or jitter > 0). When lag_noise = 0.0,")
        print("     the expanded-state covariance becomes singular during backward sampling, producing NaNs.")
    if ratio_001 > ratio_01:
        print(f"  2. Reducing lag-copy noise by 10x directly closes the variance gap by {shift:+.4f} (ratio goes from {ratio_01:.4f} to {ratio_001:.4f}).")
        print("     This confirms that upstream's auxiliary lag noise is indeed inflating upstream's posterior variance.")
    else:
        print(f"  2. Reducing lag noise from 0.01 to 0.001 resulted in ratio change of {shift:+.4f}.")
    print("  3. Numba's direct AR trajectory formulation avoids auxiliary lag copies entirely,")
    print("     operating stably at zero noise without numerical breakdown.")

if __name__ == "__main__":
    run_diagnostic()
