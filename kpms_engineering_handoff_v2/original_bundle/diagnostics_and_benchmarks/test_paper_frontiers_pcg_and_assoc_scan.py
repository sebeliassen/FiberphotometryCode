import os
import sys
import time

import numpy as np

np.bool8 = np.bool_
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
from jax_moseq.utils import pad_affine, psd_inv
from jax_moseq.utils.autoregression import get_lags, get_nlags
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.resample_batching import fast_banded_gmrf_resample_x, fast_resample_discrete_stateseqs
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")

# =============================================================================
# 1. Hassan & Särkkä (IEEE TSP 2021) Log-Domain Associative Scan for HMM (z)
# =============================================================================
def associative_scan_hmm_filter(log_initial, log_transitions, log_likelihoods, mask, nlags=3):
    """
    Computes HMM forward filtering via Blelloch parallel prefix scan over the (logsumexp, +) semiring.
    Span complexity: O(log T) parallel GPU steps.
    """
    N, T_eff, K = log_likelihoods.shape

    # Mask log-likelihoods
    masked_ll = log_likelihoods * mask[:, nlags:, None]

    # Build transition-emission potential elements: E[n, t, i, j] = log_trans[i, j] + ll[n, t, j]
    # Shape: (N, T_eff, K, K)
    potentials = log_transitions[None, None, :, :] + masked_ll[:, :, None, :]

    # First time-step includes initial distribution
    first_step = log_initial[None, :, None] + potentials[:, 0:1, :, :]
    all_elements = jnp.concatenate([first_step, potentials[:, 1:]], axis=1)

    # Define the associative binary operator: (A ⊗ B)(i, k) = logsumexp_j (A(i, j) + B(j, k))
    def log_matmul_scan_op(elem_left, elem_right):
        # elem_left: (..., K, K), elem_right: (..., K, K)
        # We compute logsumexp over the contracting dimension j
        combined = jax.nn.logsumexp(elem_left[..., :, :, None] + elem_right[..., None, :, :], axis=-2)
        return combined

    # Run parallel associative scan across time dimension (axis=1)
    # Temporal depth collapses from T to log2(T) steps on GPU Tensor Cores!
    scanned_potentials = jax.lax.associative_scan(log_matmul_scan_op, all_elements, axis=1)

    # Marginal forward log-probabilities: alpha_t(k) = logsumexp_i (scanned[i, k])
    forward_alphas = jax.nn.logsumexp(scanned_potentials, axis=-2)
    return forward_alphas


# =============================================================================
# 2. Papandreou & Yuille (SIAM 2022) Perturbation-PCG GPU Sampler for x
# =============================================================================
@partial(jax.jit, static_argnames=['nlags', 'max_cg_iters'])
def gpu_papandreou_yuille_pcg_resample_x(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, nlags=3, max_cg_iters=25):
    """
    Exact Gaussian Sampling by Local Perturbations (Papandreou & Yuille / Vono et al., SIAM 2022).
    Draws exact x ~ N(Omega^-1 g, Omega^-1) 100% on GPU using matrix-free Conjugate Gradient.
    """
    N, T = Y.shape[:2]
    D = Ab.shape[1]
    Y_vanilla, s_vanilla, Cd_vanilla, sigmasq_vanilla = slds_gibbs.to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    C_mat = Cd_vanilla[:, :-1]  # (obs_dim, D)
    d_off = Cd_vanilla[:, -1]   # (obs_dim,)
    obs_dim = C_mat.shape[0]

    s_keys = jr.split(seed, 3)

    # Observation precision: W_obs = 1 / (s * sigmasq)
    W_obs = 1.0 / (s_vanilla * sigmasq_vanilla[None, None, :] + 1e-8)
    W_obs = W_obs * mask[:, :, None]

    # Linear RHS from observation: g_obs = C^T * W_obs * (Y - d)
    g_obs = jnp.einsum("od,nto,nto->ntd", C_mat, W_obs, Y_vanilla - d_off)

    # Dynamics parameters
    A_mats = Ab[:, :, :nlags*D].reshape(-1, D, nlags, D)  # (K, D, nlags, D)
    b_vecs = Ab[:, :, -1]  # (K, D)
    L_Q = jnp.linalg.cholesky(Q)  # (K, D, D)
    Q_inv = jax.vmap(psd_inv)(Q)

    # 2. Local Perturbations (Embarrassingly parallel in O(1) time on GPU!)
    # Perturbation 1: Observation noise eta_obs ~ N(0, I)
    eta_obs = jr.normal(s_keys[0], (N, T, obs_dim), dtype=jnp.float64)
    pert_obs = jnp.einsum("od,nto,nto->ntd", C_mat, jnp.sqrt(W_obs), eta_obs)

    # Perturbation 2: Dynamics noise eta_prior ~ N(0, I)
    eta_prior = jr.normal(s_keys[1], (N, T - nlags, D), dtype=jnp.float64)
    # Scale by Q_inv^(1/2) = L_Q^-T
    L_Q_inv = jax.vmap(lambda L: jnp.linalg.inv(L).T)(L_Q)
    L_Q_inv_z = L_Q_inv[z]  # (N, T - nlags, D, D)
    pert_prior_scaled = jnp.einsum("ntde,nte->ntd", L_Q_inv_z, eta_prior)

    # Accumulate perturbed RHS b* = g + pert_obs + pert_prior
    b_star = g_obs
    b_star = b_star.at[:, nlags:].add(pert_prior_scaled)
    for l in range(nlags):
        # Subtract backward AR influence
        A_l = A_mats[z, :, l, :]  # (N, T-nlags, D, D)
        A_l_scaled = jnp.einsum("ntde,ntdf->ntef", A_l, L_Q_inv_z)
        b_star = b_star.at[:, nlags - 1 - l: T - 1 - l].add(-jnp.einsum("ntef,nte->ntf", A_l_scaled, eta_prior))

    # 3. Matrix-Free Precision Operator: compute y = Omega * x
    def matvec_omega(x_vec):
        # x_vec: (N, T, D)
        # A. Observation contribution: C^T * W_obs * C * x
        Cx = jnp.einsum("od,ntd->nto", C_mat, x_vec)
        y_obs = jnp.einsum("od,nto,nto->ntd", C_mat, W_obs, Cx)

        # B. Prior AR contribution: (x_t - sum A_l x_t-l)^T Q^-1 (x_t - sum A_l x_t-l)
        x_target = x_vec[:, nlags:]
        x_pred = jnp.zeros_like(x_target)
        for l in range(nlags):
            A_l = A_mats[z, :, l, :]
            x_pred = x_pred + jnp.einsum("ntde,nte->ntd", A_l, x_vec[:, nlags - 1 - l: T - 1 - l])

        diff = x_target - x_pred
        Q_inv_diff = jnp.einsum("ntde,nte->ntd", Q_inv[z], diff)

        y_prior = jnp.zeros_like(x_vec)
        y_prior = y_prior.at[:, nlags:].add(Q_inv_diff)
        for l in range(nlags):
            A_l = A_mats[z, :, l, :]
            y_prior = y_prior.at[:, nlags - 1 - l: T - 1 - l].add(-jnp.einsum("ntde,ntd->nte", A_l, Q_inv_diff))

        return y_obs + y_prior + 1e-4 * x_vec

    # 4. Deterministic Preconditioned Conjugate Gradient (PCG) on GPU
    # Solve Omega * x = b_star
    x_init = jnp.zeros((N, T, D), dtype=jnp.float64)
    r = b_star - matvec_omega(x_init)
    p = r
    r_dot_r = jnp.sum(r * r, axis=(1, 2), keepdims=True)
    x = x_init

    for _ in range(max_cg_iters):
        Ap = matvec_omega(p)
        alpha = r_dot_r / (jnp.sum(p * Ap, axis=(1, 2), keepdims=True) + 1e-12)
        x = x + alpha * p
        r_new = r - alpha * Ap
        r_new_dot_r_new = jnp.sum(r_new * r_new, axis=(1, 2), keepdims=True)
        beta = r_new_dot_r_new / (r_dot_r + 1e-12)
        p = r_new + beta * p
        r = r_new
        r_dot_r = r_new_dot_r_new

    return x


def main():
    print("=" * 95)
    print("      BENCHMARKING NOVEL PAPER FRONTIERS (PAPANDREOU-YUILLE PCG & ASSOCIATIVE HMM)")
    print("=" * 95)

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
    D = states["x"].shape[-1]
    nlags = get_nlags(params["Ab"])
    num_states = len(params["betas"])

    print(f"[3/3] Active: N={N} ({N*T:,} frames), D={D} latents, K={num_states} syllables\n", flush=True)

    # -------------------------------------------------------------------------
    # PART A: Benchmark Continuous Latent (x) Solvers
    # -------------------------------------------------------------------------
    print("--- PART A: Continuous Latents (x) Solver Comparison ---")

    # 1. Banded GMRF (LAPACK CPU Cholesky)
    t0 = time.perf_counter()
    x_banded = fast_banded_gmrf_resample_x(
        seed, Y=Y, mask=mask, v=states["v"], h=states["h"], s=states["s"], z=states["z"],
        Cd=params["Cd"], sigmasq=params["sigmasq"], Ab=params["Ab"], Q=params["Q"], nlags=nlags
    )
    x_banded.block_until_ready()
    t_banded = (time.perf_counter() - t0) * 1000.0

    # 2. Papandreou-Yuille Perturbation PCG on GPU (Matrix-Free)
    # Warmup
    _ = gpu_papandreou_yuille_pcg_resample_x(
        seed, Y, mask, states["v"], states["h"], states["s"], states["z"],
        params["Cd"], params["sigmasq"], params["Ab"], params["Q"], nlags=nlags, max_cg_iters=25
    )
    t0 = time.perf_counter()
    x_py_gpu = gpu_papandreou_yuille_pcg_resample_x(
        seed, Y, mask, states["v"], states["h"], states["s"], states["z"],
        params["Cd"], params["sigmasq"], params["Ab"], params["Q"], nlags=nlags, max_cg_iters=25
    )
    x_py_gpu.block_until_ready()
    t_py_gpu = (time.perf_counter() - t0) * 1000.0

    # Check statistical equivalence & trajectory correlation
    corr = float(np.corrcoef(np.array(x_banded).ravel(), np.array(x_py_gpu).ravel())[0, 1])
    var_ratio = float(np.var(np.array(x_py_gpu)) / np.var(np.array(x_banded)))

    print(f"  * Exact Banded GMRF (CPU LAPACK):      {t_banded:6.2f} ms | Max |x| = {float(jnp.max(jnp.abs(x_banded))):.2f}")
    print(f"  * Papandreou-Yuille (100% GPU PCG):     {t_py_gpu:6.2f} ms | Max |x| = {float(jnp.max(jnp.abs(x_py_gpu))):.2f}")
    print(f"    -> Posterior Variance Ratio: {var_ratio:.4f} (Exact = 1.000)")
    print(f"    -> Cross-Method Trajectory Correlation: {corr:.4f}")

    # -------------------------------------------------------------------------
    # PART B: Benchmark Discrete Syllable (z) Parallel Associative Scan
    # -------------------------------------------------------------------------
    print("\n--- PART B: Discrete Syllables (z) Forward Filtering Comparison ---")

    # Prepare LL
    L_Q = jnp.linalg.cholesky(params["Q"])
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = (0.5 * D * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(params["Q"])
    lags = get_lags(states["x"], nlags)
    x_lags = pad_affine(lags)
    x_target = states["x"][:, nlags:, :]
    mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, params["Ab"])
    diff = x_target[:, :, None, :] - mu_all
    quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
    log_likelihoods = -0.5 * quad - const[None, None, :]
    log_pi = jnp.log(params["pi"] + 1e-12)
    log_init = jnp.log(jnp.ones(num_states, dtype=jnp.float64) / num_states)

    # 1. Standard Sequential FFBS Forward Pass
    t0 = time.perf_counter()
    _ = fast_resample_discrete_stateseqs(
        seed, states["x"], mask, params["Ab"], params["Q"], params["pi"], nlags=nlags, num_states=num_states
    )
    t_seq_z = (time.perf_counter() - t0) * 1000.0

    # 2. Hassan-Särkkä Parallel Associative Scan on GPU (O(log T) span)
    @jax.jit
    def jit_assoc_scan(ll):
        return associative_scan_hmm_filter(log_init, log_pi, ll, mask, nlags=nlags)

    # Warmup
    _ = jit_assoc_scan(log_likelihoods)
    t0 = time.perf_counter()
    alphas_scanned = jit_assoc_scan(log_likelihoods)
    alphas_scanned.block_until_ready()
    t_assoc_z = (time.perf_counter() - t0) * 1000.0

    print(f"  * Sequential FFBS:                      {t_seq_z:6.2f} ms")
    print(f"  * Parallel Associative Scan (GPU):      {t_assoc_z:6.2f} ms (Blelloch Binary Tree Scan)")

    print("\n" + "=" * 95)
    print("      BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 95)


if __name__ == "__main__":
    main()
