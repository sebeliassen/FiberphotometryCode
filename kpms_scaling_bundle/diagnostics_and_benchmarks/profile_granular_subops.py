import os
import sys
import time
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)
import numpy as np

np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
import pandas as pd
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jax_moseq.utils import pad_affine, psd_inv, psd_solve
from jax_moseq.utils.autoregression import get_lags, get_nlags
from jax_moseq.utils.transitions import count_transitions, sample_hdp_transitions
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")
def profile_granular_subops(n_iters: int = 5):
    print("=" * 80)
    print("      DEEP GRANULAR SUB-OPERATION PROFILER (MICROSECOND ACCURACY)")
    jax.config.update("jax_enable_x64", True)
    # 1. Load Data
    sessions = pd.read_pickle(DIRTY_PKL)[:1]
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine="jonathan", cam="center_cam")
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, "deeplabcut")
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
    betas = model["params"]["betas"]
    Cd = model["params"]["Cd"]
    sigmasq = model["params"]["sigmasq"]
    sigmasq_loc = model["hypparams"]["cen_hypparams"]["sigmasq_loc"]
    noise_prior = model.get("noise_prior", 1.0)
    obs_hypparams = model["hypparams"]["obs_hypparams"]
    trans_hypparams = model["hypparams"]["trans_hypparams"]
    ar_hypparams = model["hypparams"]["ar_hypparams"]
    nlags = get_nlags(Ab)
    num_states = len(betas)
    print("\nStarting granular micro-benchmarking across all sub-components...\n")
    subop_timings: Dict[str, List[float]] = {}
    def bench(name, fn, iters=10):
        # Warmup
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        elif isinstance(res, (tuple, list)):
            for r in res:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            r = fn()
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
            elif isinstance(r, (tuple, list)):
                for elem in r:
                    if hasattr(elem, "block_until_ready"):
                        elem.block_until_ready()
        avg_ms = ((time.perf_counter() - t0) / iters) * 1000.0
        subop_timings[name] = avg_ms
        return avg_ms
    # -------------------------------------------------------------
    # 1. HDP Transition Breakdown
    @jax.jit
    def j_count_transitions():
        return count_transitions(num_states, z, mask)
    t_count = bench("1a. count_transitions (GPU count matrix)", j_count_transitions)
    trans_counts = j_count_transitions()
    alpha = trans_hypparams["alpha"]
    kappa = trans_hypparams["kappa"]
    gamma = trans_hypparams["gamma"]
    def py_sample_hdp():
        return sample_hdp_transitions(seed, trans_counts, betas, alpha, kappa, gamma)
    t_crf = bench("1b. sample_hdp_transitions (CPU Numba CRF + Dirichlet)", py_sample_hdp)
    # 2. AR Dynamics Breakdown
    def j_ar_suff_stats():
        mask_flat = mask[..., nlags:].reshape(-1)
        z_flat = z.reshape(-1)
        x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
        x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])
        return mask_flat, z_flat, x_in, x_out
    t_ar_prep = bench("2a. AR lag padding & flattening (GPU)", j_ar_suff_stats)
    mask_flat, z_flat, x_in, x_out = j_ar_suff_stats()
    K_0_inv = psd_inv(ar_hypparams["K_0"])
    seeds = jr.split(seed, num_states)
    def j_ar_vectorized_mniw():
        def sample_one_state(seed_k, k_idx):
            m_k = mask_flat * (z_flat == k_idx)
            x_out_m = x_out * m_k[:, None]
            S_out_out = x_out_m.T @ x_out
            S_out_in = x_out_m.T @ x_in
            x_in_m = x_in * m_k[:, None]
            S_in_in = x_in_m.T @ x_in
            K_n_inv = K_0_inv + S_in_in
            K_n = psd_inv(K_n_inv)
            M_n = psd_solve(K_n_inv.T, K_0_inv @ ar_hypparams["M_0"].T + S_out_in.T).T
            S_n = ar_hypparams["S_0"] + S_out_out + (ar_hypparams["M_0"] @ K_0_inv @ ar_hypparams["M_0"].T - M_n @ K_n_inv @ M_n.T)
            return arhmm_gibbs.sample_mniw(seed_k, ar_hypparams["nu_0"] + m_k.sum(), S_n, M_n, K_n)
        return jax.vmap(sample_one_state)(seeds, jnp.arange(num_states))
    t_ar_mniw = bench("2b. Vectorized MNIW posterior solve (GPU)", j_ar_vectorized_mniw)
    # 3. Discrete Stateseqs (z) Breakdown
    L_Q = jnp.linalg.cholesky(Q)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = 0.5 * (x.shape[-1] * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(Q)
    def j_einsum_ll():
        x_lags = pad_affine(get_lags(x, nlags))
        x_target = x[:, nlags:, :]
        mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
        diff = x_target[:, :, None, :] - mu_all
        quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
        return -0.5 * quad - const[None, None, :]
    t_z_ll = bench("3a. Tensor-Core AR Log-Likelihood einsum (GPU)", j_einsum_ll)
    log_likelihoods = j_einsum_ll()
    initial_distribution = jnp.ones(num_states) / num_states
    def j_hmm_sample():
        def sample_one_hmm(seed_i, ll_i, mask_i):
            masked_ll = ll_i * mask_i[nlags:, None]
            _, z_out = hmm_posterior_sample(seed_i, initial_distribution, pi, masked_ll)
            return z_out
        return jax.vmap(sample_one_hmm)(jr.split(seed, x.shape[0]), log_likelihoods, mask)
    t_z_hmm = bench("3b. HMM Backward/Forward Dynamic Programming (GPU)", j_hmm_sample)
    # 4. Continuous Stateseqs (x - Kalman) Breakdown
    import jax_moseq.utils.kalman as jkalman
    obs_dim = (Y.shape[-2] - 1) * Y.shape[-1]
    latent_dim = x.shape[-1]
    lds_dim = latent_dim * nlags
    m0_init = jnp.zeros(lds_dim)
    S0_init = jnp.eye(lds_dim) * 1e4
    R_dummy = jnp.ones((Y.shape[1], obs_dim))
    y_dummy = jnp.zeros((Y.shape[1], obs_dim))
    def j_kalman_emission_setup():
        return jkalman.ar_to_lds_emissions(Cd, R_dummy, y_dummy, m0_init, S0_init, nlags)
    t_x_setup = bench("4a. Kalman emission setup (ar_to_lds_emissions)", j_kalman_emission_setup)
    def j_kalman_full():
        return slds_gibbs.resample_continuous_stateseqs(
            seed, Y=Y, mask=mask, v=v, h=h, s=s, z=z, Cd=Cd, sigmasq=sigmasq, Ab=Ab, Q=Q,
            jitter=1e-3, parallel_message_passing=False
        )
    t_x_kalman = bench("4b. Serial Kalman smoother + backward sample (GPU)", j_kalman_full, iters=5)
    # 5. Pose Resampling Breakdown (h & v)
    k, d = Y.shape[-2:]
    def j_estimate_aligned():
        return slds_gibbs.estimate_aligned(x, Cd, k)
    t_est_aligned = bench("5a. estimate_aligned (Cd @ x projection)", j_estimate_aligned)
    Y_aligned = j_estimate_aligned()
    def j_heading_vmf():
        variance = s * sigmasq
        Y_cent = Y - v[..., None, :]
        S = (Y_aligned[..., :2, None] * Y_cent[..., None, :2] / variance[..., None, None]).sum(-3)
        kappa_cos = S[..., 0, 0] + S[..., 1, 1]
        kappa_sin = S[..., 0, 1] - S[..., 1, 0]
        mean_direction = jnp.stack([kappa_cos, kappa_sin], axis=-1)
        sampled_direction = slds_gibbs.sample_vonmises_fisher(seed, mean_direction)
        return slds_gibbs.vector_to_angle(sampled_direction)
    t_h_vmf = bench("5b. Heading von Mises-Fisher sampler (GPU)", j_heading_vmf)
    def j_loc_kalman():
        return slds_gibbs.resample_location(seed, Y=Y, mask=mask, x=x, h=h, s=s, Cd=Cd, sigmasq=sigmasq, sigmasq_loc=sigmasq_loc)
    t_v_loc = bench("5c. Centroid 2D Random Walk Kalman sampler (GPU)", j_loc_kalman, iters=5)
    # 6. Noise Scales (s) Breakdown
    def j_sq_error():
        return slds_gibbs.compute_squared_error(Y, x, v, h, Cd, mask)
    t_sqerr = bench("6a. compute_squared_error (Y - Cd*x - v - h)^2", j_sq_error)
    sqerr = j_sq_error()
    import jax_moseq.models.slds.gibbs as slds_base_gibbs
    def j_invgamma_scales():
        return slds_base_gibbs.resample_scales_from_sqerr(seed, sqerr, sigmasq, **obs_hypparams, s_0=noise_prior)
    t_invgamma = bench("6b. Inverse-Gamma posterior noise scales (GPU)", j_invgamma_scales)
    # Summary Table
    print("\n" + "=" * 90)
    print(f"{'DEEP GRANULAR SUB-OPERATION BREAKDOWN TABLE':^90}")
    print("=" * 90)
    print(f"{'Sub-Operation Name':<58} | {'Avg Time (ms)':<14} | {'% of Iteration':<14}")
    print("-" * 90)
    total_ms = sum(subop_timings.values())
    for name, dur_ms in subop_timings.items():
        pct = (dur_ms / total_ms) * 100.0
        print(f"{name:<58} | {dur_ms:9.2f} ms   | {pct:6.1f} %")
    print(f"{'Total Sum of Sub-Operations':<58} | {total_ms:9.2f} ms   | 100.0 %")
if __name__ == "__main__":
    profile_granular_subops()
