import os
import sys
import time

import numpy as np

np.bool8 = np.bool_
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from numba import njit, prange

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jax_moseq.utils import pad_affine, psd_inv
from jax_moseq.utils.autoregression import get_lags
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")

# =============================================================================
# 1. Tridiagonal Centroid (v) Sampler (Decoupled 2D GMRF)
# =============================================================================
@njit(fastmath=True, parallel=True)
def _assemble_and_solve_tridiagonal_v(Y, mask, Y_aligned, s, sigmasq, sigmasq_loc, N, T, k, d):
    # Tridiagonal system: main diag (row 0), sub diag (row 1)
    # v is (N, T, 2)
    v_sampled = np.zeros((N, T, d), dtype=np.float64)

    # Process variance
    q_loc_inv = 1.0 / (sigmasq_loc + 1e-8)

    for seq_i in prange(N):
        # We solve for each spatial dimension (x, y) independently!
        for dim_i in range(d):
            # Form tridiagonal precision: main diagonal ab_0, subdiagonal ab_1
            # In LAPACK lower symmetric banded: row 0 is main diag, row 1 is subdiag
            diag = np.zeros(T, dtype=np.float64)
            subdiag = np.zeros(T - 1, dtype=np.float64)
            rhs = np.zeros(T, dtype=np.float64)

            # Prior / Random Walk dynamics: v_t - v_{t-1} ~ N(0, sigmasq_loc)
            for t in range(T):
                if t == 0:
                    diag[0] += 1e-4 + q_loc_inv
                elif t == T - 1:
                    diag[T - 1] += q_loc_inv
                    diag[t - 1] += q_loc_inv
                    subdiag[t - 1] -= q_loc_inv
                else:
                    diag[t] += 2.0 * q_loc_inv
                    subdiag[t - 1] -= q_loc_inv

            # Observation terms
            for t in range(T):
                if mask[seq_i, t]:
                    obs_prec_sum = 0.0
                    obs_rhs_sum = 0.0
                    for k_i in range(k):
                        inv_var = 1.0 / (sigmasq[k_i * d + dim_i] * s[seq_i, t, k_i * d + dim_i] + 1e-8)
                        obs_prec_sum += inv_var
                        obs_rhs_sum += inv_var * (Y[seq_i, t, k_i, dim_i] - Y_aligned[seq_i, t, k_i, dim_i])
                    diag[t] += obs_prec_sum
                    rhs[t] += obs_rhs_sum

            # Thomas algorithm for Cholesky L L^T of positive definite tridiagonal
            # L is lower bidiagonal with main diagonal l_d and subdiagonal l_sub
            l_d = np.zeros(T, dtype=np.float64)
            l_sub = np.zeros(T - 1, dtype=np.float64)

            l_d[0] = np.sqrt(diag[0])
            for t in range(1, T):
                l_sub[t - 1] = subdiag[t - 1] / l_d[t - 1]
                l_d[t] = np.sqrt(max(diag[t] - l_sub[t - 1] * l_sub[t - 1], 1e-8))

            # Forward solve: L w = rhs
            w = np.zeros(T, dtype=np.float64)
            w[0] = rhs[0] / l_d[0]
            for t in range(1, T):
                w[t] = (rhs[t] - l_sub[t - 1] * w[t - 1]) / l_d[t]

            # Gaussian perturbation: eta ~ N(0, 1)
            eta = np.random.standard_normal(T)
            rhs_sample = w + eta

            # Back substitution: L^T v = w + eta
            v_sampled[seq_i, T - 1, dim_i] = rhs_sample[T - 1] / l_d[T - 1]
            for t in range(T - 2, -1, -1):
                v_sampled[seq_i, t, dim_i] = (rhs_sample[t] - l_sub[t] * v_sampled[seq_i, t + 1, dim_i]) / l_d[t]

    return v_sampled


# =============================================================================
# 2. Fast CPU HMM Forward-Filtering Backward-Sampling (FFBS)
# =============================================================================
@njit(fastmath=True, parallel=True)
def _numba_cpu_ffbs(log_likes, mask, pi_log, init_log, N, T, K, nlags):
    z_samples = np.zeros((N, T - nlags), dtype=np.int64)

    for seq_i in prange(N):
        # Forward filtering in log-space
        # alpha[t, k] = log p(z_t = k, y_{1:t})
        T_eff = T - nlags
        alpha = np.zeros((T_eff, K), dtype=np.float64)

        # t = 0
        for k in range(K):
            alpha[0, k] = init_log[k] + (log_likes[seq_i, 0, k] if mask[seq_i, nlags] else 0.0)

        # Forward pass
        for t in range(1, T_eff):
            m_t = mask[seq_i, nlags + t]
            for k_curr in range(K):
                # logsumexp over k_prev
                max_val = -1e300
                for k_prev in range(K):
                    val = alpha[t - 1, k_prev] + pi_log[k_prev, k_curr]
                    if val > max_val:
                        max_val = val
                sum_exp = 0.0
                for k_prev in range(K):
                    sum_exp += np.exp(alpha[t - 1, k_prev] + pi_log[k_prev, k_curr] - max_val)
                log_sum = max_val + np.log(sum_exp + 1e-300)
                alpha[t, k_curr] = log_sum + (log_likes[seq_i, t, k_curr] if m_t else 0.0)

        # Backward sampling
        # Sample z_{T-1} from softmax(alpha[T-1])
        max_a = -1e300
        for k in range(K):
            if alpha[T_eff - 1, k] > max_a:
                max_a = alpha[T_eff - 1, k]
        probs = np.zeros(K, dtype=np.float64)
        sum_p = 0.0
        for k in range(K):
            p = np.exp(alpha[T_eff - 1, k] - max_a)
            probs[k] = p
            sum_p += p
        probs /= sum_p

        # Draw z_{T-1}
        u = np.random.random()
        cum = 0.0
        z_t = K - 1
        for k in range(K):
            cum += probs[k]
            if u <= cum:
                z_t = k
                break
        z_samples[seq_i, T_eff - 1] = z_t

        # Backward recurrence: t = T-2 down to 0
        for t in range(T_eff - 2, -1, -1):
            next_z = z_samples[seq_i, t + 1]
            max_b = -1e300
            for k in range(K):
                val = alpha[t, k] + pi_log[k, next_z]
                if val > max_b:
                    max_b = val
            sum_b = 0.0
            for k in range(K):
                p = np.exp(alpha[t, k] + pi_log[k, next_z] - max_b)
                probs[k] = p
                sum_b += p
            probs /= sum_b

            u = np.random.random()
            cum = 0.0
            z_t = K - 1
            for k in range(K):
                cum += probs[k]
                if u <= cum:
                    z_t = k
                    break
            z_samples[seq_i, t] = z_t

    return z_samples


def main():
    print("=" * 80)
    print("      BENCHMARKING TIER-1 FRONTIERS: TRIDIAGONAL v AND FAST HMM z")
    print("=" * 80)

    # 1. Load Data
    sessions = pd.read_pickle(DIRTY_PKL)[:1]
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine="jonathan", cam="center_cam")
    coordinates, confidences, _ = kpms.load_keypoints(keypoint_paths, "deeplabcut")
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}
    filtered_coords, filtered_confs, _ = filter_with_masks(
        coordinates, confidences, all_masks, SCRATCH_DIR,
        return_video_frame_indexes=True, downsample_rate=2,
        **kpms.load_config(SCRATCH_DIR)
    )
    clamp_conf_nonneg(filtered_confs)
    data, _ = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(SCRATCH_DIR))
    data = {k: v[:2] for k, v in data.items()}
    pca = kpms.fit_pca(**data, **kpms.load_config(SCRATCH_DIR))

    seed = jr.PRNGKey(42)
    init_config = kpms.load_config(SCRATCH_DIR)
    model = batched_init_model_by_video(data, pca, init_config, params_subset_size=2, batch_size=20, seed=seed)
    model = kpms.update_hypparams(model, kappa=6000.0)

    x = model["states"]["x"]
    z = model["states"]["z"]
    v = model["states"]["v"]
    h = model["states"]["h"]
    s = model["states"]["s"]
    Y = data["Y"]
    mask = data["mask"]
    Ab = model["params"]["Ab"]
    Q = model["params"]["Q"]
    pi = model["params"]["pi"]
    Cd = model["params"]["Cd"]
    sigmasq = model["params"]["sigmasq"]
    sigmasq_loc = float(model["hypparams"]["cen_hypparams"]["sigmasq_loc"])

    N, T, k, d = Y.shape
    nlags = 3
    num_states = 100

    # -------------------------------------------------------------
    # Test Centroid v (Kalman vs Tridiagonal)
    # -------------------------------------------------------------
    print("\n--- 1. Centroid Location (v) Benchmarks ---")

    # Baseline JAX Kalman Location
    t0 = time.perf_counter()
    for _ in range(5):
        v_kalman = slds_gibbs.resample_location(seed, Y=Y, mask=mask, x=x, h=h, s=s, Cd=Cd, sigmasq=sigmasq, sigmasq_loc=sigmasq_loc)
        v_kalman.block_until_ready()
    t_v_kalman = (time.perf_counter() - t0) / 5.0 * 1000.0
    print(f"  * Baseline JAX 2D Kalman Filter : {t_v_kalman:7.2f} ms")

    # Fast Tridiagonal GMRF
    Y_np = np.array(Y)
    mask_np = np.array(mask, dtype=bool)
    Y_aligned_np = np.array(slds_gibbs.estimate_aligned(x, Cd, k))
    # Rotate aligned coords by h
    h_np = np.array(h)
    cos_h = np.cos(h_np)[:, :, None]
    sin_h = np.sin(h_np)[:, :, None]
    Y_rot_np = np.zeros_like(Y_aligned_np)
    Y_rot_np[..., 0] = cos_h * Y_aligned_np[..., 0] - sin_h * Y_aligned_np[..., 1]
    Y_rot_np[..., 1] = sin_h * Y_aligned_np[..., 0] + cos_h * Y_aligned_np[..., 1]
    s_np = np.array(s)
    sigmasq_np = np.array(sigmasq)

    # Warmup Numba
    _ = _assemble_and_solve_tridiagonal_v(Y_np, mask_np, Y_rot_np, s_np, sigmasq_np, sigmasq_loc, N, T, k, d)

    t0 = time.perf_counter()
    for _ in range(10):
        v_tridiag = _assemble_and_solve_tridiagonal_v(Y_np, mask_np, Y_rot_np, s_np, sigmasq_np, sigmasq_loc, N, T, k, d)
    t_v_tridiag = (time.perf_counter() - t0) / 10.0 * 1000.0
    print(f"  * Fast Tridiagonal GMRF Solver  : {t_v_tridiag:7.2f} ms ({t_v_kalman/t_v_tridiag:5.1f}x speedup!)")

    # -------------------------------------------------------------
    # Test Discrete Stateseqs z (JAX GPU vs Numba CPU)
    # -------------------------------------------------------------
    print("\n--- 2. Discrete Stateseqs (z) Benchmarks ---")

    L_Q = jnp.linalg.cholesky(Q)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = 0.5 * (x.shape[-1] * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(Q)

    x_lags = pad_affine(get_lags(x, nlags))
    x_target = x[:, nlags:, :]
    mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
    diff = x_target[:, :, None, :] - mu_all
    quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
    log_likelihoods = -0.5 * quad - const[None, None, :]

    initial_distribution = jnp.ones(num_states) / num_states

    # Option A: Baseline Dynamax JAX Sequential FFBS
    @jax.jit
    def j_hmm_sample():
        def sample_one_hmm(seed_i, ll_i, mask_i):
            masked_ll = ll_i * mask_i[nlags:, None]
            _, z_out = hmm_posterior_sample(seed_i, initial_distribution, pi, masked_ll)
            return z_out
        return jax.vmap(sample_one_hmm)(jr.split(seed, x.shape[0]), log_likelihoods, mask)

    _ = j_hmm_sample()
    t0 = time.perf_counter()
    for _ in range(5):
        z_jax = j_hmm_sample()
        z_jax.block_until_ready()
    t_z_jax = (time.perf_counter() - t0) / 5.0 * 1000.0
    print(f"  * Option A: JAX Sequential FFBS (GPU) : {t_z_jax:7.2f} ms")

    # Option B: Numba Multi-threaded C FFBS (CPU)
    ll_np = np.array(log_likelihoods, dtype=np.float64)
    pi_log_np = np.log(np.array(pi, dtype=np.float64) + 1e-300)
    init_log_np = np.log(np.array(initial_distribution, dtype=np.float64) + 1e-300)

    # Warmup Numba
    _ = _numba_cpu_ffbs(ll_np, mask_np, pi_log_np, init_log_np, N, T, num_states, nlags)

    t0 = time.perf_counter()
    for _ in range(10):
        z_cpu = _numba_cpu_ffbs(ll_np, mask_np, pi_log_np, init_log_np, N, T, num_states, nlags)
    t_z_cpu = (time.perf_counter() - t0) / 10.0 * 1000.0
    print(f"  * Option B: Compiled Parallel FFBS (CPU) : {t_z_cpu:7.2f} ms ({t_z_jax/t_z_cpu:5.1f}x speedup!)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
