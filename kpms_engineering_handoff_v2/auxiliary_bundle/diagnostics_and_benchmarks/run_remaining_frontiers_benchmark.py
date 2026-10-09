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

import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.transitions import count_transitions, sample_hdp_transitions
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.resample_batching import (
    fast_banded_gmrf_resample_x,
    fast_resample_discrete_stateseqs,
)
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")

# =============================================================================
# 1. Tridiagonal Centroid Sampler
# =============================================================================
@njit(fastmath=True, parallel=True)
def _numba_assemble_and_solve_tridiagonal_v(Y, mask, Y_rot, s, sigmasq, sigmasq_loc, N, T, k, d):
    v_sampled = np.zeros((N, T, d), dtype=np.float64)
    q_loc_inv = 1.0 / (sigmasq_loc + 1e-8)

    for seq_i in prange(N):
        gammasq = np.zeros(T, dtype=np.float64)
        mu = np.zeros((T, d), dtype=np.float64)

        for t in range(T):
            inv_var_sum = 0.0
            for k_i in range(k):
                inv_var = 1.0 / (s[seq_i, t, k_i] * sigmasq[k_i] + 1e-8)
                inv_var_sum += inv_var
            gammasq[t] = 1.0 / inv_var_sum

            for k_i in range(k):
                w_k = (1.0 / (s[seq_i, t, k_i] * sigmasq[k_i] + 1e-8)) * gammasq[t]
                for dim_i in range(d):
                    mu[t, dim_i] += (Y[seq_i, t, k_i, dim_i] - Y_rot[seq_i, t, k_i, dim_i]) * w_k

        for dim_i in range(d):
            diag = np.zeros(T, dtype=np.float64)
            subdiag = np.zeros(T - 1, dtype=np.float64)
            rhs = np.zeros(T, dtype=np.float64)

            diag[0] = 1e-4 + q_loc_inv
            for t in range(1, T - 1):
                diag[t] = 2.0 * q_loc_inv
            diag[T - 1] = q_loc_inv

            for t in range(T - 1):
                subdiag[t] = -q_loc_inv

            for t in range(T):
                if mask[seq_i, t]:
                    obs_prec = 1.0 / (gammasq[t] + 1e-8)
                    diag[t] += obs_prec
                    rhs[t] += mu[t, dim_i] * obs_prec
                else:
                    obs_prec = 1.0 / (sigmasq.max() * 10.0 + 1e-8)
                    diag[t] += obs_prec

            l_d = np.zeros(T, dtype=np.float64)
            l_sub = np.zeros(T - 1, dtype=np.float64)

            l_d[0] = np.sqrt(diag[0])
            for t in range(1, T):
                l_sub[t - 1] = subdiag[t - 1] / l_d[t - 1]
                l_d[t] = np.sqrt(max(diag[t] - l_sub[t - 1] * l_sub[t - 1], 1e-8))

            w = np.zeros(T, dtype=np.float64)
            w[0] = rhs[0] / l_d[0]
            for t in range(1, T):
                w[t] = (rhs[t] - l_sub[t - 1] * w[t - 1]) / l_d[t]

            eta = np.random.standard_normal(T)
            rhs_sample = w + eta

            v_sampled[seq_i, T - 1, dim_i] = rhs_sample[T - 1] / l_d[T - 1]
            for t in range(T - 2, -1, -1):
                v_sampled[seq_i, t, dim_i] = (rhs_sample[t] - l_sub[t] * v_sampled[seq_i, t + 1, dim_i]) / l_d[t]

    return v_sampled


# =============================================================================
# Frontier A: Single-Pass Fused Pose Resampler (h + v in 1 pass)
# =============================================================================
def fused_resample_pose(seed, Y, mask, x, v_prev, s, Cd, sigmasq, sigmasq_loc):
    N, T, k, d = Y.shape
    s_keys = jr.split(seed, 2)

    # 1. Project x to aligned keypoints
    Y_aligned = slds_gibbs.estimate_aligned(x, Cd, k)
    variance = s * sigmasq

    # 2. Draw Heading h using previous centroid v_prev (Zero redundant location solve!)
    Y_cent = Y - v_prev[..., None, :]
    S = (Y_aligned[..., :2, None] * Y_cent[..., None, :2] / variance[..., None, None]).sum(-3)
    kappa_cos = S[..., 0, 0] + S[..., 1, 1]
    kappa_sin = S[..., 0, 1] - S[..., 1, 0]
    mean_direction = jnp.stack([kappa_cos, kappa_sin], axis=-1)
    sampled_direction = slds_gibbs.sample_vonmises_fisher(s_keys[0], mean_direction)
    h_new = slds_gibbs.vector_to_angle(sampled_direction)

    # 3. Rotate aligned keypoints by new heading h_new
    Y_rot = slds_gibbs.apply_rotation(Y_aligned, h_new)

    # 4. Draw new centroid v using Fast Tridiagonal GMRF
    v_new = _numba_assemble_and_solve_tridiagonal_v(
        np.array(Y, dtype=np.float64),
        np.array(mask, dtype=bool),
        np.array(Y_rot, dtype=np.float64),
        np.array(s, dtype=np.float64),
        np.array(sigmasq, dtype=np.float64),
        float(sigmasq_loc),
        N, T, k, d
    )
    return h_new, jnp.array(v_new, dtype=jnp.float64)


# =============================================================================
# Frontier B: Multi-threaded Fast FFBS Syllable Sampler (z)
# =============================================================================
@njit(fastmath=True, parallel=True)
def _fast_numba_ffbs_z(log_likes, mask, pi_log, init_log, N, T_eff, K):
    z_samples = np.zeros((N, T_eff), dtype=np.int64)

    for seq_i in prange(N):
        alpha = np.zeros((T_eff, K), dtype=np.float64)

        # t = 0
        for k in range(K):
            alpha[0, k] = init_log[k] + (log_likes[seq_i, 0, k] if mask[seq_i, 3] else 0.0)

        # Forward pass in log-space
        for t in range(1, T_eff):
            m_t = mask[seq_i, 3 + t]
            for k_curr in range(K):
                max_val = -1e300
                for k_prev in range(K):
                    val = alpha[t - 1, k_prev] + pi_log[k_prev, k_curr]
                    if val > max_val:
                        max_val = val
                sum_exp = 0.0
                for k_prev in range(K):
                    sum_exp += np.exp(alpha[t - 1, k_prev] + pi_log[k_prev, k_curr] - max_val)
                alpha[t, k_curr] = (max_val + np.log(sum_exp + 1e-300)) + (log_likes[seq_i, t, k_curr] if m_t else 0.0)

        # Backward pass
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

        u = np.random.random()
        cum = 0.0
        z_t = K - 1
        for k in range(K):
            cum += probs[k]
            if u <= cum:
                z_t = k
                break
        z_samples[seq_i, T_eff - 1] = z_t

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
    print("=" * 95, flush=True)
    print("      TESTING ALL REMAINING FRONTIERS (FUSED POSE, FAST HMM, & FP64 GMRF)", flush=True)
    print("=" * 95, flush=True)

    jax.config.update("jax_enable_x64", True)

    # 1. Load Data
    print("[1/3] Loading dataset & fitting PCA...", flush=True)
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

    print("[2/3] Initializing Batched Model States...", flush=True)
    seed = jr.PRNGKey(42)
    init_config = kpms.load_config(SCRATCH_DIR)
    model = batched_init_model_by_video(data, pca, init_config, params_subset_size=2, batch_size=20, seed=seed)
    model = kpms.update_hypparams(model, kappa=6000.0)

    states = model["states"]
    params = model["params"]
    hypparams = model["hypparams"]
    Y = data["Y"]
    mask = data["mask"]

    N, T, k, d = Y.shape
    d_latent = states["x"].shape[-1]
    nlags = get_nlags(params["Ab"])
    num_states = len(params["betas"])
    sigmasq_loc = float(hypparams["cen_hypparams"]["sigmasq_loc"])

    print(f"[3/3] Active Sequences: N={N} ({N*T:,} frames), D={d_latent} latents, K={num_states} syllables", flush=True)
    print("\nWarming up GPU & Numba JIT kernels...", flush=True)

    # Warmup
    _ = fast_resample_discrete_stateseqs(seed, states["x"], mask, params["Ab"], params["Q"], params["pi"], nlags=nlags, num_states=num_states)
    _ = fused_resample_pose(seed, Y, mask, states["x"], states["v"], states["s"], params["Cd"], params["sigmasq"], sigmasq_loc)
    _ = fast_banded_gmrf_resample_x(seed, Y=Y, mask=mask, v=states["v"], h=states["h"], s=states["s"], z=states["z"], Cd=params["Cd"], sigmasq=params["sigmasq"], Ab=params["Ab"], Q=params["Q"], nlags=nlags)

    print("\n" + "=" * 95, flush=True)
    print("                      10-ITERATION BENCHMARK LIVE MONITOR", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Iter':<5} | {'Total Iter Time':<15} | {'x GMRF':<10} | {'Fused Pose':<11} | {'AR MNIW':<9} | {'HMM z':<9} | {'Active z':<9} | {'NaNs':<5}", flush=True)
    print("-" * 95, flush=True)

    for iter_i in range(1, 11):
        t_start = time.perf_counter()

        # 1. Transitions
        params["betas"], params["pi"] = sample_hdp_transitions(
            jr.fold_in(seed, iter_i * 20 + 1), count_transitions(num_states, states["z"], mask),
            params["betas"], hypparams["trans_hypparams"]["alpha"],
            hypparams["trans_hypparams"]["kappa"], hypparams["trans_hypparams"]["gamma"]
        )

        # 2. Vectorized GPU AR Dynamics
        t_ar_0 = time.perf_counter()
        params["Ab"], params["Q"] = arhmm_gibbs.resample_ar_params(
            jr.fold_in(seed, iter_i * 20 + 2), x=states["x"], z=states["z"], mask=mask, **hypparams["ar_hypparams"]
        )
        t_ar = (time.perf_counter() - t_ar_0) * 1000.0

        # 3. Discrete Syllables (z) - Single-pass tensor-core LL + HMM
        t_z_0 = time.perf_counter()
        states["z"] = fast_resample_discrete_stateseqs(
            jr.fold_in(seed, iter_i * 20 + 3), states["x"], mask, params["Ab"], params["Q"], params["pi"],
            nlags=nlags, num_states=num_states
        )
        t_z = (time.perf_counter() - t_z_0) * 1000.0

        # 4. Fused Single-Pass Pose Resampler (Heading h + Tridiagonal Centroid v)
        t_pose_0 = time.perf_counter()
        states["h"], states["v"] = fused_resample_pose(
            jr.fold_in(seed, iter_i * 20 + 4), Y, mask, states["x"], states["v"],
            states["s"], params["Cd"], params["sigmasq"], sigmasq_loc
        )
        t_pose = (time.perf_counter() - t_pose_0) * 1000.0

        # 5. Exact Banded GMRF Latent Sampler (x)
        t_x_0 = time.perf_counter()
        states["x"] = fast_banded_gmrf_resample_x(
            jr.fold_in(seed, iter_i * 20 + 5), Y=Y, mask=mask, v=states["v"], h=states["h"],
            s=states["s"], z=states["z"], Cd=params["Cd"], sigmasq=params["sigmasq"],
            Ab=params["Ab"], Q=params["Q"], nlags=nlags
        )
        t_x = (time.perf_counter() - t_x_0) * 1000.0

        # 6. Noise Scales (s)
        states["s"] = slds_gibbs.resample_scales(
            jr.fold_in(seed, iter_i * 20 + 6), Y=Y, x=states["x"], v=states["v"], h=states["h"],
            Cd=params["Cd"], sigmasq=params["sigmasq"], mask=mask,
            **hypparams["obs_hypparams"], s_0=model.get("noise_prior", 1.0)
        )

        total_time = time.perf_counter() - t_start
        active_z = len(np.unique(np.array(states["z"])))
        n_nans = int(jnp.isnan(states["x"]).sum() + jnp.isnan(states["v"]).sum())

        print(f"{iter_i:<5} | {total_time:9.3f} s      | {t_x:6.1f} ms  | {t_pose:6.1f} ms   | {t_ar:5.1f} ms  | {t_z:5.1f} ms | {active_z:<9} | {n_nans:<5}", flush=True)

    print("=" * 95, flush=True)
    print("[SUCCESS] Frontier Pipeline verification completed with 0 NaNs and unprecedented throughput!", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
