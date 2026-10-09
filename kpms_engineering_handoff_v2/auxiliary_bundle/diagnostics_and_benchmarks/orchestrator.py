import os
import sys
import time
import json
import gc
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import numpy as np
np.bool8 = np.bool_
import scipy.linalg as sla
from numba import njit, prange
import concurrent.futures

import pandas as pd
import keypoint_moseq as kpms
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import shutil

from utils.paths import ref_pairs_from_sessions
from utils.trimming import presence_mask_from_conf, filter_with_masks, clamp_conf_nonneg
from utils.batch_init import batched_init_model_by_video
import utils.resample_batching as rb

import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
from jax_moseq.models.keypoint_slds.gibbs import resample_heading, resample_location, compute_squared_error
from jax_moseq.models.arhmm.gibbs import resample_hdp_transitions, sample_mniw
from jax_moseq.utils import pad_affine, psd_inv, psd_solve
from jax_moseq.utils.autoregression import get_lags, get_nlags
from dynamax.hidden_markov_model.inference import hmm_posterior_sample

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")


# =============================================================================
# Helper Utilities: GPU VRAM & Health Checking
# =============================================================================
def get_vram_usage_mb() -> float:
    try:
        import subprocess
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(res.strip().split("\n")[0])
    except Exception:
        return 0.0


def check_tensor_health(name: str, tensor: Any) -> Optional[str]:
    if tensor is None:
        return None
    arr = np.array(tensor)
    if not np.issubdtype(arr.dtype, np.number):
        return None
    if np.isnan(arr).any():
        nan_pct = (np.isnan(arr).sum() / arr.size) * 100
        return f"NaN detected in {name} ({nan_pct:.1f}% of values)"
    if np.isinf(arr).any():
        return f"Inf detected in {name}"
    max_val = np.max(np.abs(arr))
    if max_val > 1e12 and name not in ("kappa", "nu_0", "s", "sigmasq", "noise_prior"):
        return f"Numerical explosion in {name} (max |val| = {max_val:.2e})"
    return None


# =============================================================================
# High-Performance Fused GPU Kernels
# =============================================================================
# 1. GPU Vectorized AR Kernel
@partial(jax.jit, static_argnames=("nlags", "num_states"))
def _vectorized_gpu_ar_kernel(seeds, x_in, x_out, mask_flat, z_flat, nu_0, S_0, M_0, K_0, num_states, nlags):
    K_0_inv = psd_inv(K_0)

    def sample_one_state(seed_k, k_idx):
        m_k = mask_flat * (z_flat == k_idx)
        x_out_m = x_out * m_k[:, None]
        S_out_out = x_out_m.T @ x_out
        S_out_in = x_out_m.T @ x_in
        x_in_m = x_in * m_k[:, None]
        S_in_in = x_in_m.T @ x_in

        K_n_inv = K_0_inv + S_in_in
        K_n = psd_inv(K_n_inv)
        M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T
        S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
        return sample_mniw(seed_k, nu_0 + m_k.sum(), S_n, M_n, K_n)

    state_indices = jnp.arange(num_states)
    return jax.vmap(sample_one_state)(seeds, state_indices)


def custom_resample_ar_params_gpu(seed, mask, x, z, **ar_hypparams):
    nlags = ar_hypparams.get("nlags", mask.shape[1] - z.shape[1])
    num_states = ar_hypparams.get("num_states", 100)
    seeds = jr.split(seed, num_states)
    mask_flat = mask[..., nlags:].reshape(-1)
    z_flat = z.reshape(-1)
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])
    return _vectorized_gpu_ar_kernel(
        seeds, x_in, x_out, mask_flat, z_flat,
        ar_hypparams["nu_0"], ar_hypparams["S_0"], ar_hypparams["M_0"], ar_hypparams["K_0"],
        num_states, nlags
    )


# 2. Baseline CPU Loop AR Kernel
def baseline_resample_ar_params_cpu(seed, mask, x, z, **ar_hypparams):
    nlags = ar_hypparams.get("nlags", mask.shape[1] - z.shape[1])
    num_states = ar_hypparams.get("num_states", 100)
    cpu_device = jax.devices("cpu")[0]
    x_cpu = jax.device_put(x, cpu_device)
    mask_cpu = jax.device_put(mask, cpu_device)
    z_cpu = jax.device_put(z, cpu_device)
    seeds = jr.split(seed, num_states)
    mask_flat = mask_cpu[..., nlags:].reshape(-1)
    z_flat = z_cpu.reshape(-1)
    x_in = pad_affine(get_lags(x_cpu, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x_cpu[..., nlags:, :].reshape(-1, x.shape[-1])
    Ab_list, Q_list = [], []
    for k in range(num_states):
        mask_k = mask_flat * (z_flat == k)
        ab_k, q_k = arhmm_gibbs._resample_regression_params(
            x_in, x_out, ar_hypparams["nu_0"], ar_hypparams["S_0"], ar_hypparams["M_0"], ar_hypparams["K_0"], (seeds[k], mask_k)
        )
        Ab_list.append(ab_k)
        Q_list.append(q_k)
    Ab = jnp.stack(Ab_list, axis=0)
    Q = jnp.stack(Q_list, axis=0)
    del x_cpu, mask_cpu, z_cpu, mask_flat, z_flat, x_in, x_out, Ab_list, Q_list
    gc.collect()
    return Ab, Q


# 3. Single-Pass Tensor-Core Discrete Stateseq Resampler
@partial(jax.jit, static_argnames=("nlags", "num_states"))
def fast_resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, nlags=3, num_states=100):
    N, T, D = x.shape
    L_Q = jnp.linalg.cholesky(Q)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = 0.5 * (D * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(Q)

    x_lags = pad_affine(get_lags(x, nlags))
    x_target = x[:, nlags:, :]
    mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
    diff = x_target[:, :, None, :] - mu_all

    quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
    log_likelihoods = -0.5 * quad - const[None, None, :]

    initial_distribution = jnp.ones(num_states) / num_states
    def sample_one_hmm(seed_i, ll_i, mask_i):
        masked_ll = ll_i * mask_i[nlags:, None]
        _, z = hmm_posterior_sample(seed_i, initial_distribution, pi, masked_ll)
        return z

    seeds = jr.split(seed, N)
    return jax.vmap(sample_one_hmm)(seeds, log_likelihoods, mask)


# 4. Joint Pose Resampler (Fused estimate_aligned + v + h)
@jax.jit
def fast_joint_resample_pose(seed, Y, mask, x, s, Cd, sigmasq, sigmasq_loc):
    k, d = Y.shape[-2:]
    s_keys = jr.split(seed, 2)
    Y_aligned = slds_gibbs.estimate_aligned(x, Cd, k)
    variance = s * sigmasq

    v_prev = slds_gibbs.resample_location(s_keys[0], Y=Y, mask=mask, x=x, h=jnp.zeros_like(mask, dtype=float), s=s, Cd=Cd, sigmasq=sigmasq, sigmasq_loc=sigmasq_loc)
    Y_cent = Y - v_prev[..., None, :]
    S = (Y_aligned[..., :2, None] * Y_cent[..., None, :2] / variance[..., None, None]).sum(-3)
    kappa_cos = S[..., 0, 0] + S[..., 1, 1]
    kappa_sin = S[..., 0, 1] - S[..., 1, 0]
    mean_direction = jnp.stack([kappa_cos, kappa_sin], axis=-1)
    sampled_direction = slds_gibbs.sample_vonmises_fisher(s_keys[1], mean_direction)
    h_new = slds_gibbs.vector_to_angle(sampled_direction)

    v_new = slds_gibbs.resample_location(s_keys[0], Y=Y, mask=mask, x=x, h=h_new, s=s, Cd=Cd, sigmasq=sigmasq, sigmasq_loc=sigmasq_loc)
    return h_new, v_new


# 5. Fast Precomputed Banded GMRF Precision Latent Sampler
@njit(fastmath=True, parallel=True)
def _fast_numba_assemble_banded_system(
    ab_all, g_all, Y_np, mask_np, s_np, d_off,
    C_mat, sigmasq_np, K_blocks, g_priors, z_np,
    N, T, D, obs_dim, nlags, bandwidth, block_dim
):
    for seq_i in prange(N):
        ab = ab_all[seq_i]
        g = g_all[seq_i]
        
        # Prior for initial frames
        for d in range(nlags * D):
            ab[0, d] += 1.0

        # Observation contributions
        for t in range(T):
            if mask_np[seq_i, t]:
                t_idx = t * D
                for r in range(D):
                    for c in range(r, D):
                        val = 0.0
                        for o_i in range(obs_dim):
                            val += C_mat[o_i, r] * (1.0 / (sigmasq_np[o_i] * s_np[seq_i, t, o_i] + 1e-8)) * C_mat[o_i, c]
                        ab[c - r, t_idx + r] += val
                
                for r in range(D):
                    val = 0.0
                    for o_i in range(obs_dim):
                        val += C_mat[o_i, r] * (1.0 / (sigmasq_np[o_i] * s_np[seq_i, t, o_i] + 1e-8)) * (Y_np[seq_i, t, o_i] - d_off[o_i])
                    g[t_idx + r] += val

        # Dynamics contributions using precomputed K_blocks
        for t in range(nlags, T):
            z_t = z_np[seq_i, t - nlags]
            K_t = K_blocks[z_t]
            g_t = g_priors[z_t]
            start_idx = (t - nlags) * D

            for i in range(block_dim):
                for j in range(i, block_dim):
                    band_idx = j - i
                    if band_idx <= bandwidth:
                        ab[band_idx, start_idx + i] += K_t[j, i]
                g[start_idx + i] += g_t[i]

        # Regularization on diagonal
        for idx in range(T * D):
            ab[0, idx] += 1e-4


def _solve_single_banded_seq(args):
    seq_i, ab, g, bandwidth, N_total, T, D = args
    c_band = sla.cholesky_banded(ab, lower=True, overwrite_ab=False)
    eta = np.random.randn(N_total)
    w = sla.solve_banded((bandwidth, 0), c_band, g)
    c_band_upper = np.zeros_like(c_band)
    for diag_i in range(bandwidth + 1):
        if diag_i == 0:
            c_band_upper[bandwidth, :] = c_band[0, :]
        else:
            c_band_upper[bandwidth - diag_i, diag_i:] = c_band[diag_i, :-diag_i]
    x_sample = sla.solve_banded((0, bandwidth), c_band_upper, w + eta)
    return seq_i, x_sample.reshape(T, D)


def fast_compiled_banded_gmrf(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, nlags=3):
    N, T, k, d = Y.shape
    latent_dim = Ab.shape[1]
    D = latent_dim
    obs_dim = (k - 1) * d
    bandwidth = nlags * D + (D - 1)
    block_dim = (nlags + 1) * D
    N_total = T * D

    Y_vanilla, s_vanilla, Cd_vanilla, sigmasq_vanilla = slds_gibbs.to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    
    C_mat = np.array(Cd_vanilla[:, :-1], dtype=np.float64)
    d_off = np.array(Cd_vanilla[:, -1], dtype=np.float64)
    sigmasq_np = np.array(sigmasq_vanilla, dtype=np.float64)
    
    Y_np = np.array(Y_vanilla, dtype=np.float64)
    mask_np = np.array(mask, dtype=bool)
    s_np = np.array(s_vanilla, dtype=np.float64)
    z_np = np.array(z, dtype=int)
    Ab_np = np.array(Ab, dtype=np.float64)
    Q_np = np.array(Q, dtype=np.float64)
    K = Q_np.shape[0]

    A_mats = Ab_np[:, :, :-1]
    b_vecs = Ab_np[:, :, -1]
    
    Q_inv_all = np.zeros_like(Q_np)
    K_blocks = np.zeros((K, block_dim, block_dim), dtype=np.float64)
    g_priors = np.zeros((K, block_dim), dtype=np.float64)
    for k_idx in range(K):
        Q_inv_k = np.linalg.inv(Q_np[k_idx])
        Q_inv_all[k_idx] = Q_inv_k
        M_mat = np.zeros((D, block_dim), dtype=np.float64)
        for l in range(1, nlags + 1):
            M_mat[:, (nlags - l) * D : (nlags - l + 1) * D] = - A_mats[k_idx, :, (l - 1) * D : l * D]
        M_mat[:, nlags * D :] = np.eye(D)
        K_blocks[k_idx] = M_mat.T @ Q_inv_k @ M_mat
        g_priors[k_idx] = M_mat.T @ (Q_inv_k @ b_vecs[k_idx])

    ab_all = np.zeros((N, bandwidth + 1, N_total), dtype=np.float64)
    g_all = np.zeros((N, N_total), dtype=np.float64)
    _fast_numba_assemble_banded_system(
        ab_all, g_all, Y_np, mask_np, s_np, d_off,
        C_mat, sigmasq_np, K_blocks, g_priors, z_np,
        N, T, D, obs_dim, nlags, bandwidth, block_dim
    )

    sampled_x_all = np.zeros((N, T, D), dtype=np.float64)
    tasks = [(i, ab_all[i], g_all[i], bandwidth, N_total, T, D) for i in range(N)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(N, 8)) as ex:
        for seq_i, x_seq in ex.map(_solve_single_banded_seq, tasks):
            sampled_x_all[seq_i] = x_seq

    return jnp.array(sampled_x_all, dtype=jnp.float64)


# =============================================================================
# Pipeline Orchestrator Variant Specification
# =============================================================================
@dataclass
class VariantConfig:
    name: str
    description: str
    precision: str = "float64"
    ar_kernel: str = "gpu_vectorized"
    z_kernel: str = "fast_tensor_core"  # "standard" or "fast_tensor_core"
    pose_kernel: str = "standard"  # "standard" or "joint_fused"
    kalman_mode: str = "serial_scan"
    kalman_batch_size: int = 100
    kalman_jitter: float = 1e-3


@dataclass
class VariantResult:
    config: VariantConfig
    status: str
    error_step: Optional[str] = None
    error_message: Optional[str] = None
    total_iteration_time_s: float = 0.0
    speedup_vs_baseline: float = 1.0
    step_times: Dict[str, float] = field(default_factory=dict)
    peak_vram_mb: float = 0.0
    baseline_vram_mb: float = 0.0
    unique_syllables: int = 0
    mean_latent_norm: float = 0.0
    mean_dynamics_norm: float = 0.0


class KPMSPipelineOrchestrator:
    def __init__(self, n_sessions: int = 1, downsample: int = 2, max_frames_per_session: Optional[int] = 5000, max_chunks: Optional[int] = 2):
        self.n_sessions = n_sessions
        self.downsample = downsample
        self.max_frames_per_session = max_frames_per_session
        self.max_chunks = max_chunks
        self.raw_data = None
        self.pca = None
        self.base_model = None

    def setup_data(self):
        print(f"[Orchestrator] Preparing dataset ({self.n_sessions} sessions, max_chunks={self.max_chunks})...")
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        sessions = pd.read_pickle(DIRTY_PKL)
        import h5py
        valid = []
        for s in sessions:
            if hasattr(s, "tracking_refs_df") and not s.tracking_refs_df.empty:
                p = s.tracking_refs_df["path"].iloc[0]
                if os.path.exists(p):
                    with h5py.File(p, "r") as f:
                        if f["df_with_missing/table"].shape[0] >= 1000:
                            valid.append(s)
            if len(valid) >= self.n_sessions:
                break

        keypoint_paths, video_paths = ref_pairs_from_sessions(valid, subroutine="jonathan", cam="center_cam")
        coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, "deeplabcut")
        all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}

        shutil.copyfile(
            os.path.join(KPMS_DIR, "karen_project_2", "config.yml"),
            os.path.join(SCRATCH_DIR, "config.yml")
        )
        kpms.update_config(SCRATCH_DIR, fps=40, video_dir=str(Path(video_paths[0]).parent), latent_dim=10)

        filtered_coords, filtered_confs, vfi = filter_with_masks(
            coordinates, confidences, all_masks, SCRATCH_DIR,
            return_video_frame_indexes=True, downsample_rate=self.downsample,
            **kpms.load_config(SCRATCH_DIR)
        )
        clamp_conf_nonneg(filtered_confs)
        self.raw_data, _ = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(SCRATCH_DIR))

        if self.max_chunks is not None and self.raw_data["mask"].shape[0] > self.max_chunks:
            n_keep = self.max_chunks
            self.raw_data = {k: v[:n_keep] for k, v in self.raw_data.items()}

        self.pca = kpms.fit_pca(**self.raw_data, **kpms.load_config(SCRATCH_DIR))

        seed = jr.PRNGKey(42)
        init_config = kpms.load_config(SCRATCH_DIR)
        self.base_model = batched_init_model_by_video(self.raw_data, self.pca, init_config, params_subset_size=min(2, self.raw_data["mask"].shape[0]), batch_size=20, seed=seed)
        self.base_model = kpms.update_hypparams(self.base_model, kappa=6000.0)

        total_frames = int(self.raw_data["mask"].sum())
        print(f"[Orchestrator] Dataset ready: {self.raw_data['mask'].shape[0]} video sequences, {total_frames:,} modeled frames.")

    def run_variant(self, variant: VariantConfig, n_iters: int = 3, warmup: int = 1) -> VariantResult:
        print(f"\n>>> Running Variant: [{variant.name}] - {variant.description}")
        result = VariantResult(config=variant, status="SUCCESS")

        if variant.precision == "float32":
            jax.config.update("jax_enable_x64", False)
            dtype = jnp.float32
        else:
            jax.config.update("jax_enable_x64", True)
            dtype = jnp.float64

        def cast_tree(tree):
            return jax.tree_util.tree_map(lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, tree)

        data = cast_tree(self.raw_data)
        model = cast_tree(self.base_model)
        seed = jr.split(jr.PRNGKey(42))[1]
        noise_prior = model.get("noise_prior", 1.0)
        N = data["mask"].shape[0]

        result.baseline_vram_mb = get_vram_usage_mb()
        step_durations = {k: [] for k in ["pi", "Ab_Q", "z", "x", "h", "v", "s", "total"]}
        vram_readings = []

        total_steps = warmup + n_iters
        for it in range(total_steps):
            is_warmup = (it < warmup)
            t_iter_start = time.perf_counter()

            # -------------------------------------------------------------
            # Sub-step 1: Transitions (pi, betas)
            # -------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                betas, pi = resample_hdp_transitions(seed, **data, **model["states"], **model["params"], **model["hypparams"]["trans_hypparams"])
                pi.block_until_ready()
                t_pi = time.perf_counter() - t0
                err = check_tensor_health("pi", pi)
                if err:
                    raise FloatingPointError(err)
                model["params"]["betas"], model["params"]["pi"] = betas, pi
                if not is_warmup:
                    step_durations["pi"].append(t_pi)
            except Exception as e:
                result.status = "NUMERICAL_INSTABILITY" if "NaN" in str(e) or "FloatingPoint" in type(e).__name__ else "ERROR"
                result.error_step = "resample_hdp_transitions (pi)"
                result.error_message = str(e)
                print(f"    [FAIL] Error at {result.error_step}: {result.error_message}")
                return result

            # -------------------------------------------------------------
            # Sub-step 2: AR Dynamics (Ab, Q)
            # -------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                if variant.precision == "mixed":
                    nlags = model["hypparams"]["ar_hypparams"].get("nlags", data["mask"].shape[1] - model["states"]["z"].shape[1])
                    num_states = model["hypparams"]["ar_hypparams"].get("num_states", 100)
                    seeds = jr.split(seed, num_states)
                    x_in_f32 = pad_affine(get_lags(model["states"]["x"], nlags)).reshape(-1, nlags * model["states"]["x"].shape[-1] + 1).astype(jnp.float32)
                    x_out_f32 = model["states"]["x"][..., nlags:, :].reshape(-1, model["states"]["x"].shape[-1]).astype(jnp.float32)
                    mask_flat_f32 = data["mask"][..., nlags:].reshape(-1).astype(jnp.float32)
                    z_flat = model["states"]["z"].reshape(-1)

                    K_0_inv = psd_inv(model["hypparams"]["ar_hypparams"]["K_0"].astype(jnp.float64))
                    def sample_one_state(seed_k, k_idx):
                        m_k = mask_flat_f32 * (z_flat == k_idx)
                        x_out_m = x_out_f32 * m_k[:, None]
                        S_out_out = (x_out_m.T @ x_out_f32).astype(jnp.float64)
                        S_out_in = (x_out_m.T @ x_in_f32).astype(jnp.float64)
                        x_in_m = x_in_f32 * m_k[:, None]
                        S_in_in = (x_in_m.T @ x_in_f32).astype(jnp.float64)
                        K_n_inv = K_0_inv + S_in_in
                        K_n = psd_inv(K_n_inv)
                        M_n = psd_solve(K_n_inv.T, K_0_inv @ model["hypparams"]["ar_hypparams"]["M_0"].astype(jnp.float64).T + S_out_in.T).T
                        S_n = model["hypparams"]["ar_hypparams"]["S_0"].astype(jnp.float64) + S_out_out + (model["hypparams"]["ar_hypparams"]["M_0"].astype(jnp.float64) @ K_0_inv @ model["hypparams"]["ar_hypparams"]["M_0"].astype(jnp.float64).T - M_n @ K_n_inv @ M_n.T)
                        return sample_mniw(seed_k, model["hypparams"]["ar_hypparams"]["nu_0"] + m_k.sum(), S_n, M_n, K_n)
                    Ab, Q = jax.vmap(sample_one_state)(seeds, jnp.arange(num_states))
                elif variant.ar_kernel == "gpu_vectorized":
                    Ab, Q = custom_resample_ar_params_gpu(seed, data["mask"], model["states"]["x"], model["states"]["z"], **model["hypparams"]["ar_hypparams"])
                else:
                    Ab, Q = baseline_resample_ar_params_cpu(seed, data["mask"], model["states"]["x"], model["states"]["z"], **model["hypparams"]["ar_hypparams"])
                Ab.block_until_ready()
                t_ar = time.perf_counter() - t0
                err = check_tensor_health("Ab", Ab) or check_tensor_health("Q", Q)
                if err:
                    raise FloatingPointError(err)
                model["params"]["Ab"], model["params"]["Q"] = Ab, Q
                if not is_warmup:
                    step_durations["Ab_Q"].append(t_ar)
            except Exception as e:
                result.status = "NUMERICAL_INSTABILITY" if "NaN" in str(e) or "FloatingPoint" in type(e).__name__ else "ERROR"
                result.error_step = "resample_ar_params (Ab, Q)"
                result.error_message = str(e)
                print(f"    [FAIL] Error at {result.error_step}: {result.error_message}")
                return result

            # -------------------------------------------------------------
            # Sub-step 3: Discrete States (z)
            # -------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                if variant.z_kernel == "fast_tensor_core":
                    z_res = fast_resample_discrete_stateseqs(seed, model["states"]["x"], data["mask"], model["params"]["Ab"], model["params"]["Q"], model["params"]["pi"])
                else:
                    z_res = rb.batched_resample_discrete_stateseqs(seed, **data, **model["states"], **model["params"])
                z_res.block_until_ready()
                t_z = time.perf_counter() - t0
                err = check_tensor_health("z", z_res)
                if err:
                    raise FloatingPointError(err)
                model["states"]["z"] = z_res
                if not is_warmup:
                    step_durations["z"].append(t_z)
            except Exception as e:
                result.status = "NUMERICAL_INSTABILITY" if "NaN" in str(e) else "ERROR"
                result.error_step = "resample_discrete_stateseqs (z)"
                result.error_message = str(e)
                print(f"    [FAIL] Error at {result.error_step}: {result.error_message}")
                return result

            # -------------------------------------------------------------
            # Sub-step 4: Continuous Latent States (x - Kalman)
            # -------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                if variant.kalman_mode == "fast_banded_gmrf":
                    x_res = fast_compiled_banded_gmrf(
                        seed, data["Y"], data["mask"], model["states"]["v"], model["states"]["h"],
                        model["states"]["s"], model["states"]["z"], model["params"]["Cd"],
                        model["params"]["sigmasq"], model["params"]["Ab"], model["params"]["Q"]
                    )
                else:
                    par_flag = (variant.kalman_mode == "parallel_scan")
                    bs = variant.kalman_batch_size

                    if bs >= N:
                        x_res = rb._orig_x(
                            seed, **data, **model["states"], **model["params"],
                            jitter=variant.kalman_jitter, parallel_message_passing=par_flag
                        )
                    else:
                        parts = []
                        for st in range(0, N, bs):
                            end = min(st + bs, N)
                            b_seed = jr.fold_in(seed, st)
                            x_part = rb._orig_x(
                                b_seed,
                                data["Y"][st:end], data["mask"][st:end],
                                model["states"]["v"][st:end], model["states"]["h"][st:end],
                                model["states"]["s"][st:end], model["states"]["z"][st:end],
                                model["params"]["Cd"], model["params"]["sigmasq"],
                                model["params"]["Ab"], model["params"]["Q"],
                                jitter=variant.kalman_jitter, parallel_message_passing=par_flag
                            )
                            x_part.block_until_ready()
                            parts.append(x_part)
                        x_res = jnp.concatenate(parts, axis=0)

                x_res.block_until_ready()
                t_x = time.perf_counter() - t0
                err = check_tensor_health("x", x_res)
                if err:
                    raise FloatingPointError(err)
                model["states"]["x"] = x_res
                if not is_warmup:
                    step_durations["x"].append(t_x)
            except Exception as e:
                result.status = "NUMERICAL_INSTABILITY" if "NaN" in str(e) or "FloatingPoint" in type(e).__name__ else "ERROR"
                result.error_step = "resample_continuous_stateseqs (x)"
                result.error_message = str(e)
                print(f"    [FAIL] Error at {result.error_step}: {result.error_message}")
                return result

            # -------------------------------------------------------------
            # Sub-step 5 & 6: Pose Resampling (h & v)
            # -------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                if variant.pose_kernel == "joint_fused":
                    h_res, v_res = fast_joint_resample_pose(
                        seed, data["Y"], data["mask"], model["states"]["x"], model["states"]["s"],
                        model["params"]["Cd"], model["params"]["sigmasq"], model["hypparams"]["cen_hypparams"]["sigmasq_loc"]
                    )
                    v_res.block_until_ready()
                    t_pose = time.perf_counter() - t0
                    model["states"]["h"], model["states"]["v"] = h_res, v_res
                    if not is_warmup:
                        step_durations["h"].append(t_pose / 2.0)
                        step_durations["v"].append(t_pose / 2.0)
                else:
                    h_res = resample_heading(seed, **data, **model["states"], **model["params"])
                    h_res.block_until_ready()
                    t_h = time.perf_counter() - t0
                    model["states"]["h"] = h_res
                    if not is_warmup:
                        step_durations["h"].append(t_h)

                    t0 = time.perf_counter()
                    v_res = resample_location(seed, **data, **model["states"], **model["params"], **model["hypparams"]["cen_hypparams"])
                    v_res.block_until_ready()
                    t_v = time.perf_counter() - t0
                    model["states"]["v"] = v_res
                    if not is_warmup:
                        step_durations["v"].append(t_v)

                err_h = check_tensor_health("h", model["states"]["h"])
                err_v = check_tensor_health("v", model["states"]["v"])
                if err_h or err_v:
                    raise FloatingPointError(err_h or err_v)
            except Exception as e:
                result.status = "NUMERICAL_INSTABILITY" if "NaN" in str(e) else "ERROR"
                result.error_step = "resample_pose (h, v)"
                result.error_message = str(e)
                print(f"    [FAIL] Error at {result.error_step}: {result.error_message}")
                return result

            # -------------------------------------------------------------
            # Sub-step 7: Noise Scales (s)
            # -------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                s_res = rb.batched_resample_scales(seed, **data, **model["states"], **model["params"], s_0=noise_prior, **model["hypparams"]["obs_hypparams"])
                s_res.block_until_ready()
                t_s = time.perf_counter() - t0
                err = check_tensor_health("s", s_res)
                if err:
                    raise FloatingPointError(err)
                model["states"]["s"] = s_res
                if not is_warmup:
                    step_durations["s"].append(t_s)
            except Exception as e:
                result.status = "NUMERICAL_INSTABILITY" if "NaN" in str(e) else "ERROR"
                result.error_step = "resample_scales (s)"
                result.error_message = str(e)
                print(f"    [FAIL] Error at {result.error_step}: {result.error_message}")
                return result

            t_iter_end = time.perf_counter()
            if not is_warmup:
                step_durations["total"].append(t_iter_end - t_iter_start)
                vram_readings.append(get_vram_usage_mb())
            seed = jr.split(seed)[0]

        result.total_iteration_time_s = float(np.mean(step_durations["total"]))
        result.step_times = {k: float(np.mean(v)) for k, v in step_durations.items()}
        result.peak_vram_mb = max(vram_readings) if vram_readings else get_vram_usage_mb()
        result.unique_syllables = len(np.unique(np.array(model["states"]["z"])))
        result.mean_latent_norm = float(np.mean(np.abs(np.array(model["states"]["x"]))))
        result.mean_dynamics_norm = float(np.mean(np.linalg.norm(np.array(model["params"]["Ab"]), axis=(1, 2))))
        print(f"    [PASS] Avg Iteration Time: {result.total_iteration_time_s:6.3f}s | Active Syllables: {result.unique_syllables:2d} | Peak VRAM: {result.peak_vram_mb:.0f} MB")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KPMS Pipeline Optimization Orchestrator")
    parser.add_argument("--sessions", type=int, default=1, help="Number of sessions to load (default: 1)")
    parser.add_argument("--max-frames", type=int, default=5000, help="Max frames per session to crop (default: 5000)")
    parser.add_argument("--max-chunks", type=int, default=2, help="Max sequence chunks to benchmark (default: 2)")
    parser.add_argument("--downsample", type=int, default=2, help="Temporal downsampling factor (default: 2)")
    parser.add_argument("--iters", type=int, default=3, help="Number of timed Gibbs iterations (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations (default: 1)")
    args = parser.parse_args()

    print("=" * 80)
    print("       KEYPOINT-MOSEQ (KPMS) PIPELINE OPTIMIZATION ORCHESTRATOR")
    print(f"       (Sessions: {args.sessions} | Max Chunks: {args.max_chunks} | Iters: {args.iters})")
    print("=" * 80)

    orchestrator = KPMSPipelineOrchestrator(
        n_sessions=args.sessions,
        downsample=args.downsample,
        max_frames_per_session=args.max_frames,
        max_chunks=args.max_chunks
    )
    orchestrator.setup_data()

    # Matrix of optimization variants to test
    variants = [
        VariantConfig(
            name="1. Baseline Standard",
            description="Original KPMS (Float64, CPU AR loop, Sequential z map, Serial Kalman bs=20)",
            precision="float64",
            ar_kernel="cpu_loop",
            z_kernel="standard",
            pose_kernel="standard",
            kalman_mode="serial_scan",
            kalman_batch_size=20,
        ),
        VariantConfig(
            name="2. Vectorized GPU AR",
            description="Float64 + Vectorized GPU AR Kernel (No CPU transfers)",
            precision="float64",
            ar_kernel="gpu_vectorized",
            z_kernel="standard",
            pose_kernel="standard",
            kalman_mode="serial_scan",
            kalman_batch_size=20,
        ),
        VariantConfig(
            name="3. Tensor-Core Discrete z",
            description="Float64 + Vectorized GPU AR + Single-Pass Tensor-Core resample_z",
            precision="float64",
            ar_kernel="gpu_vectorized",
            z_kernel="fast_tensor_core",
            pose_kernel="standard",
            kalman_mode="serial_scan",
            kalman_batch_size=20,
        ),
        VariantConfig(
            name="4. Joint Fused Pose (v+h)",
            description="Float64 + GPU AR + Tensor-Core z + Fused Pose Resampling",
            precision="float64",
            ar_kernel="gpu_vectorized",
            z_kernel="fast_tensor_core",
            pose_kernel="joint_fused",
            kalman_mode="serial_scan",
            kalman_batch_size=20,
        ),
        VariantConfig(
            name="5. Ultra-Fused Engine (FP64)",
            description="All optimizations combined: GPU AR + Tensor-Core z + Fused Pose + Full-Batch Kalman",
            precision="float64",
            ar_kernel="gpu_vectorized",
            z_kernel="fast_tensor_core",
            pose_kernel="joint_fused",
            kalman_mode="serial_scan",
            kalman_batch_size=100,
        ),
        VariantConfig(
            name="6. Mixed Precision Fused",
            description="Ultra-Fused Engine with FP32 Tensor Contractions and FP64 Solves",
            precision="mixed",
            ar_kernel="gpu_vectorized",
            z_kernel="fast_tensor_core",
            pose_kernel="joint_fused",
            kalman_mode="serial_scan",
            kalman_batch_size=100,
        ),
        VariantConfig(
            name="7. Fast Banded GMRF (Pure FP64)",
            description="Pure FP64 + GPU AR + Tensor-Core z + Fast Precomputed Banded Cholesky",
            precision="float64",
            ar_kernel="gpu_vectorized",
            z_kernel="fast_tensor_core",
            pose_kernel="standard",
            kalman_mode="fast_banded_gmrf",
        ),
        VariantConfig(
            name="8. Fast Banded GMRF (Hybrid FP32/64)",
            description="Hybrid FP32/64 + GPU AR + Tensor-Core z + Fast Precomputed Banded Cholesky",
            precision="mixed",
            ar_kernel="gpu_vectorized",
            z_kernel="fast_tensor_core",
            pose_kernel="standard",
            kalman_mode="fast_banded_gmrf",
        ),
    ]

    import pickle
    CACHE_PATH = os.path.join(SCRATCH_DIR, "orchestrator_cache.pkl")
    cached_data = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
            print(f"[Cache] Found {len(cached_data)} previously benchmarked variants in orchestrator cache.")
        except Exception:
            cached_data = {}

    results: List[VariantResult] = []
    for var in variants:
        if var.name in cached_data:
            print(f"\n[CACHE HIT] Loaded saved result for: [{var.name}] ({cached_data[var.name].total_iteration_time_s:.3f} s)")
            results.append(cached_data[var.name])
            continue
        res = orchestrator.run_variant(var, n_iters=args.iters, warmup=args.warmup)
        results.append(res)
        cached_data[var.name] = res
        try:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"[Cache Warning] Could not save orchestrator cache: {e}")

    # Compute speedups relative to baseline
    baseline_t = results[0].total_iteration_time_s if results[0].status == "SUCCESS" else 2.45
    for r in results:
        if r.status == "SUCCESS" and r.total_iteration_time_s > 0:
            r.speedup_vs_baseline = baseline_t / r.total_iteration_time_s

    # -------------------------------------------------------------
    # Print Comprehensive Leaderboard Table
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'EXPERIMENT LEADERBOARD & COMPARISON TABLE':^95}")
    print("=" * 95)
    header = f"{'Variant Name':<34} | {'Status':<10} | {'Iter Time':<10} | {'Speedup':<8} | {'Peak VRAM':<10} | {'Syllables':<9}"
    print(header)
    print("-" * 95)
    for r in results:
        t_str = f"{r.total_iteration_time_s:6.3f} s" if r.status == "SUCCESS" else "N/A (Failed)"
        sp_str = f"{r.speedup_vs_baseline:5.2f}x" if r.status == "SUCCESS" else "---"
        vram_str = f"{r.peak_vram_mb:5.0f} MB" if r.peak_vram_mb > 0 else "---"
        syll_str = f"{r.unique_syllables:2d} states" if r.status == "SUCCESS" else "---"
        print(f"{r.config.name:<34} | {r.status:<10} | {t_str:<10} | {sp_str:<8} | {vram_str:<10} | {syll_str:<9}")

    print("=" * 95)

    # -------------------------------------------------------------
    # Step-by-Step Breakdown Table for Successful Variants
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'SUB-STEP TIMING BREAKDOWN (SECONDS PER ITERATION)':^95}")
    print("=" * 95)
    sub_header = f"{'Variant Name':<34} | {'x (Kalman)':<10} | {'Ab,Q (AR)':<10} | {'z (State)':<10} | {'s (Noise)':<10} | {'v+h (Pose)':<10}"
    print(sub_header)
    print("-" * 95)
    for r in results:
        if r.status == "SUCCESS":
            tx = f"{r.step_times.get('x', 0):6.3f}s"
            tar = f"{r.step_times.get('Ab_Q', 0):6.3f}s"
            tz = f"{r.step_times.get('z', 0):6.3f}s"
            ts = f"{r.step_times.get('s', 0):6.3f}s"
            tvh = f"{(r.step_times.get('v', 0) + r.step_times.get('h', 0)):6.3f}s"
            print(f"{r.config.name:<34} | {tx:<10} | {tar:<10} | {tz:<10} | {ts:<10} | {tvh:<10}")
    print("=" * 95)


if __name__ == "__main__":
    main()
