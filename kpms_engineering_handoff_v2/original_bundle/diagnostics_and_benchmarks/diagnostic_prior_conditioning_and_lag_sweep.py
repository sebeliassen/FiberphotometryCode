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

def build_kalman_sampler(lag_noise, use_conditioned_prior=True):
    """
    Constructs an isolated, diagnostic-only Upstream Kalman continuous state sampler.
    - If use_conditioned_prior is True: passes m0_, S0_ returned by ar_to_lds_emissions
      with in_axes mapping (0, 0, 0, 0, 0, 0, na, ...).
    - If use_conditioned_prior is False: passes unconditioned m0, S0
      matching verbatim the upstream installed code in jax_moseq.
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

        if use_conditioned_prior:
            in_axes = (0, 0, 0, 0, 0, 0, na, na, na, na, na, 0, na, na)
            prior_m, prior_S = m0_, S0_
        else:
            in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
            prior_m, prior_S = m0, S0

        x = mixed_map(
            partial(kalman_sample, jitter=0.0, parallel=parallel_message_passing),
            in_axes,
        )(
            jr.split(seed, n_recordings),
            y_,
            mask_,
            z,
            prior_m,
            prior_S,
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

    return resample_stateseqs


def run_comprehensive_diagnostic():
    print("=" * 85)
    print("  AUDIT: INITIAL PRIOR CONDITIONING (m0, S0 vs m0_, S0_) & LAG NOISE SWEEP")
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

    N_DRAWS = 50
    print(f"Slice: N={N}, T={T}, Keypoints={k}, D={D}, All-valid frames={int(np.sum(mask_np))}/{N*T}")
    print(f"Sampling budget: {N_DRAWS} draws per condition, jitter=0.0 throughout.")

    # 1. Numba Sampler
    print("\n[Condition 1] Current Numba Sampler (Direct AR, 0 lag noise, jitter=0.0)...", flush=True)
    numba_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    numba_fn = pure_numba_resample_continuous_stateseqs
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(1000 + i)
        x = numba_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=0.0)
        numba_draws[i] = np.array(x)
    print(f"  Done in {time.time() - t0:.2f}s.")

    # 2. Upstream Installed (unconditioned m0, S0, lag_noise=0.010)
    print("\n[Condition 2] Upstream Installed (Unconditioned m0, S0, lag_noise=0.010)...", flush=True)
    kalman_uncond_fn = build_kalman_sampler(lag_noise=0.010, use_conditioned_prior=False)
    k_uncond_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(2000 + i)
        x = kalman_uncond_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q)
        k_uncond_draws[i] = np.array(x)
    print(f"  Done in {time.time() - t0:.2f}s.")

    # 3. Upstream Intended (conditioned m0_, S0_, lag_noise=0.010)
    print("\n[Condition 3] Upstream Intended (Conditioned m0_, S0_, lag_noise=0.010)...", flush=True)
    kalman_cond_01_fn = build_kalman_sampler(lag_noise=0.010, use_conditioned_prior=True)
    k_cond_01_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(3000 + i)
        x = kalman_cond_01_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q)
        k_cond_01_draws[i] = np.array(x)
    print(f"  Done in {time.time() - t0:.2f}s.")

    # 4. Upstream Intended (conditioned m0_, S0_, lag_noise=0.001)
    print("\n[Condition 4] Upstream Intended (Conditioned m0_, S0_, lag_noise=0.001)...", flush=True)
    kalman_cond_001_fn = build_kalman_sampler(lag_noise=0.001, use_conditioned_prior=True)
    k_cond_001_draws = np.zeros((N_DRAWS, N, T, D), dtype=np.float64)
    t0 = time.time()
    for i in range(N_DRAWS):
        seed = jr.PRNGKey(4000 + i)
        x = kalman_cond_001_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q)
        k_cond_001_draws[i] = np.array(x)
    print(f"  Done in {time.time() - t0:.2f}s.")

    # 5. Upstream Intended (conditioned m0_, S0_, lag_noise=0.000)
    print("\n[Condition 5] Upstream Intended (Conditioned m0_, S0_, lag_noise=0.000)...", flush=True)
    kalman_cond_0_fn = build_kalman_sampler(lag_noise=0.000, use_conditioned_prior=True)
    k0_failed = False
    k0_err = None
    try:
        seed = jr.PRNGKey(5000)
        x = kalman_cond_0_fn(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q)
        if np.isnan(np.array(x)).any():
            k0_failed = True
            k0_err = "NaNs detected from singular covariance matrix in backward sampling"
    except Exception as e:
        k0_failed = True
        k0_err = f"{type(e).__name__}: {str(e)}"
    print(f"  Numerical failure detected: {k0_failed} ({k0_err})")

    # Variances (ddof=1)
    var_numba = np.var(numba_draws, axis=0, ddof=1)
    var_uncond = np.var(k_uncond_draws, axis=0, ddof=1)
    var_cond_01 = np.var(k_cond_01_draws, axis=0, ddof=1)
    var_cond_001 = np.var(k_cond_001_draws, axis=0, ddof=1)

    pvar_numba = float(np.mean(var_numba[mask_np]))
    pvar_uncond = float(np.mean(var_uncond[mask_np]))
    pvar_cond_01 = float(np.mean(var_cond_01[mask_np]))
    pvar_cond_001 = float(np.mean(var_cond_001[mask_np]))

    ratio_uncond = pvar_numba / pvar_uncond
    ratio_cond_01 = pvar_numba / pvar_cond_01
    ratio_cond_001 = pvar_numba / pvar_cond_001

    # Early frames variance (frames 0..4) vs Rest of sequence (frames 5..399)
    early_numba = float(np.mean(var_numba[:, :5]))
    early_uncond = float(np.mean(var_uncond[:, :5]))
    early_cond_01 = float(np.mean(var_cond_01[:, :5]))
    early_cond_001 = float(np.mean(var_cond_001[:, :5]))

    late_numba = float(np.mean(var_numba[:, 5:]))
    late_uncond = float(np.mean(var_uncond[:, 5:]))
    late_cond_01 = float(np.mean(var_cond_01[:, 5:]))
    late_cond_001 = float(np.mean(var_cond_001[:, 5:]))

    print("\n" + "=" * 90)
    print(f"{'QUANTITATIVE AUDIT RESULTS: PRIOR CONDITIONING & LAG NOISE':^90}")
    print("=" * 90)

    print("\n[A] OVERALL POOLED POSTERIOR VARIANCE (ALL FRAMES):")
    print(f"  1. Numba Banded Cholesky (Exact AR, 0 lag noise):          {pvar_numba:.6f}  (Baseline)")
    print(f"  2. Upstream Installed (Unconditioned m0,S0, lag=0.010):   {pvar_uncond:.6f}  --> Ratio (Numba/Installed): {ratio_uncond:.4f} (Deficit: {(1.0-ratio_uncond)*100:.1f}%)")
    print(f"  3. Upstream Intended (Conditioned m0_,S0_, lag=0.010):     {pvar_cond_01:.6f}  --> Ratio (Numba/Intended):  {ratio_cond_01:.4f} (Deficit: {(1.0-ratio_cond_01)*100:.1f}%)")
    print(f"  4. Upstream Intended (Conditioned m0_,S0_, lag=0.001):     {pvar_cond_001:.6f}  --> Ratio (Numba/Intended):  {ratio_cond_001:.4f} (Deficit: {(1.0-ratio_cond_001)*100:.1f}%)")
    print(f"  5. Upstream Intended (Conditioned m0_,S0_, lag=0.000):     NUMERICAL BREAKDOWN ({k0_err})")

    print("\n[B] TEMPORAL BREAKDOWN: INITIAL BOUNDARY (FRAMES 0-4) vs STEADY STATE (FRAMES 5-399):")
    print(f"  {'Condition':<40} | {'Frames 0-4 Var':<14} | {'Frames 5-399 Var':<16} | {'Steady Ratio':<12}")
    print("-" * 90)
    print(f"  {'Numba Banded Cholesky':<40} | {early_numba:<14.5f} | {late_numba:<16.5f} | {'1.0000':<12}")
    print(f"  {'Upstream Installed (Unconditioned m0, S0)':<40} | {early_uncond:<14.5f} | {late_uncond:<16.5f} | {late_numba/late_uncond:<12.4f}")
    print(f"  {'Upstream Intended (Conditioned m0_, S0_)':<40} | {early_cond_01:<14.5f} | {late_cond_01:<16.5f} | {late_numba/late_cond_01:<12.4f}")
    print(f"  {'Upstream Intended (10x reduced lag noise)':<40} | {early_cond_001:<14.5f} | {late_cond_001:<16.5f} | {late_numba/late_cond_001:<12.4f}")
    print("=" * 90)

    print("\n[C] PER-DIMENSION RATIOS (NUMBA / UPSTREAM INTENDED, lag=0.010):")
    dim_numba = np.mean(var_numba, axis=(0, 1))
    dim_cond_01 = np.mean(var_cond_01, axis=(0, 1))
    dim_uncond = np.mean(var_uncond, axis=(0, 1))
    header = f"{'Dim':<5} | {'Numba Var':<11} | {'Upstream Intended':<18} | {'Ratio Intended':<14} | {'Upstream Installed':<18} | {'Ratio Installed':<15}"
    print(header)
    print("-" * 90)
    for d_idx in range(D):
        r_int = dim_numba[d_idx] / dim_cond_01[d_idx]
        r_ins = dim_numba[d_idx] / dim_uncond[d_idx]
        print(f"D{d_idx:<4} | {dim_numba[d_idx]:<11.5f} | {dim_cond_01[d_idx]:<18.5f} | {r_int:<14.4f} | {dim_uncond[d_idx]:<18.5f} | {r_ins:<15.4f}")
    print("=" * 90)

if __name__ == "__main__":
    run_comprehensive_diagnostic()
