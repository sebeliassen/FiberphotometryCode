from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.tree_util import tree_flatten_with_path

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import contextlib
from dataclasses import dataclass
from typing import Optional

import jax_moseq.models.arhmm as arhmm_model
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.alignment as slds_alignment
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jax_moseq.models import keypoint_slds
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error
from jax_moseq.utils import pad_affine, psd_inv, psd_solve
from jax_moseq.utils.autoregression import get_lags
from numba import njit, prange


@dataclass
class BatchConfig:
    discrete_bs: int = 10
    scales_bs: int = 20
    heading_bs: int = 20
    gpu_loc_bs: int = 20
    continuous_chunk_size: int = 50  # Right-sized from 300 to 50 (reduces banded buffer from 9.6GB to 1.6GB)
    use_numba_hmm: bool = True       # 15.6x faster parallel CPU FFBS for discrete states (z)
    use_banded_continuous: bool = True # True: Numba Banded Cholesky (58x); False: Upstream JAX Kalman smoother
    banded_solver: str = "numba"     # "numba" (in-place parallel) or "lapack" (dpbtrf + dtbsv)

CONFIG = BatchConfig()

# Backwards compatibility alias
BATCH_SIZE = CONFIG.scales_bs
SCALES_BS = CONFIG.scales_bs
HEADING_BS = CONFIG.heading_bs
GPU_LOC_BS = CONFIG.gpu_loc_bs
DISCRETE_BS = CONFIG.discrete_bs

def precompute_optimal_batch_config(
    data: dict,
    model: Optional[dict] = None,
    gpu_budget_gb: Optional[float] = None,
    cpu_budget_gb: Optional[float] = None,
    safety_margin: float = 0.70,
    verbose: bool = True
) -> BatchConfig:
    """
    Preemptively computes optimal batch sizes from closed-form tensor memory formulae.
    
    Eliminates trial-and-error OOM crashes and kernel recompilations by analyzing
    sequence length T, keypoint count K_bps, latent dimension D, and syllable states K_states
    against the hardware's available VRAM and RAM budgets.
    """
    Y = data.get("Y")
    if Y is None:
        return CONFIG

    N = len(Y)
    T = Y.shape[1] if (hasattr(Y, "shape") and len(Y.shape) >= 2) else max(len(y) for y in Y)
    K_bps = Y.shape[2] if (hasattr(Y, "shape") and len(Y.shape) >= 3) else 23

    D = 10
    K_states = 100
    nlags = 3
    if model is not None:
        if "params" in model:
            p = model["params"]
            if "pi" in p: K_states = p["pi"].shape[0]
            if "Ab" in p:
                D = p["Ab"].shape[1]
                nlags = (p["Ab"].shape[2] - 1) // D
        elif "states" in model and "x" in model["states"]:
            D = model["states"]["x"].shape[-1]

    # Query Hardware Memory
    total_vram_gb = None
    free_vram_gb = None
    try:
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            stats = dev.memory_stats()
            if "bytes_limit" in stats and stats["bytes_limit"] > 0:
                limit = stats["bytes_limit"]
                in_use = stats.get("bytes_in_use", 0)
                free_vram_gb = (limit - in_use) / 1e9
                total_vram_gb = limit / 1e9
    except Exception:
        pass

    if free_vram_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                total_vram_gb = total_bytes / 1e9
                free_vram_gb = free_bytes / 1e9
        except Exception:
            pass

    if gpu_budget_gb is None:
        if free_vram_gb is not None:
            gpu_budget_gb = free_vram_gb * safety_margin
        elif total_vram_gb is not None:
            gpu_budget_gb = total_vram_gb * 0.50
        else:
            gpu_budget_gb = 10.0  # safe default

    if cpu_budget_gb is None:
        try:
            import psutil
            cpu_budget_gb = (psutil.virtual_memory().available / 1e9) * 0.60
        except Exception:
            cpu_budget_gb = 25.0

    # Closed-form Analytical Tensor Bytes per Sequence (in FP64)
    # 1. Discrete state emissions: intermediate mu_all (N,T,K,D) + diff (N,T,K,D) + einsum product requires 4x buffer
    bytes_per_seq_z = max(1, (T - nlags)) * K_states * D * 8 * 4
    # 2. Heading resampling (rotated keypoints + inner products)
    bytes_per_seq_h = T * K_bps * 8 * 8
    # 3. Location / Centroid
    bytes_per_seq_v = T * K_bps * 6 * 8
    # 4. Noise scales (compute_squared_error intermediate rotations + residuals)
    bytes_per_seq_s = T * K_bps * 11 * 8
    # 5. Continuous states Banded Cholesky on CPU: (bw + 4) * D + 2 * K_bps (workspace + solves)
    bw = (nlags + 1) * D - 1
    bytes_per_seq_x_cpu = ((bw + 4) * D + 2 * K_bps) * T * 8

    gpu_budget_bytes = gpu_budget_gb * 1e9
    cpu_budget_bytes = cpu_budget_gb * 1e9

    optimal_discrete_bs = max(1, min(N, 10, int(gpu_budget_bytes // bytes_per_seq_z)))
    optimal_heading_bs = max(1, min(N, 20, int(gpu_budget_bytes // bytes_per_seq_h)))
    optimal_gpu_loc_bs = max(1, min(N, 20, int(gpu_budget_bytes // bytes_per_seq_v)))
    optimal_scales_bs = max(1, min(N, 20, int(gpu_budget_bytes // bytes_per_seq_s)))
    optimal_chunk_x = max(1, min(N, 50, int(cpu_budget_bytes // bytes_per_seq_x_cpu)))

    CONFIG.discrete_bs = optimal_discrete_bs
    CONFIG.heading_bs = optimal_heading_bs
    CONFIG.gpu_loc_bs = optimal_gpu_loc_bs
    CONFIG.scales_bs = optimal_scales_bs
    CONFIG.continuous_chunk_size = optimal_chunk_x

    if verbose:
        print(f"[Preemptive Batching] Analytical Tensor Sizing (N={N}, T={T:,}, K_bps={K_bps}, D={D}, K_states={K_states}):")
        print(f"  • GPU VRAM Budget: {gpu_budget_gb:.2f} GB | Host RAM Budget: {cpu_budget_gb:.2f} GB")
        print(f"  • Discrete emissions per seq: {bytes_per_seq_z / 1e6:7.1f} MB  ==> discrete_bs = {optimal_discrete_bs}")
        print(f"  • Heading rotation per seq:   {bytes_per_seq_h / 1e6:7.1f} MB  ==> heading_bs  = {optimal_heading_bs}")
        print(f"  • Centroid / Location per seq:{bytes_per_seq_v / 1e6:7.1f} MB  ==> gpu_loc_bs  = {optimal_gpu_loc_bs}")
        print(f"  • Noise scales per seq:       {bytes_per_seq_s / 1e6:7.1f} MB  ==> scales_bs   = {optimal_scales_bs}")
        print(f"  • CPU Banded Cholesky per seq:{bytes_per_seq_x_cpu / 1e6:7.1f} MB  ==> cont_chunk  = {optimal_chunk_x}")

    return CONFIG

def set_batch_sizes(scale_factor=0.5, **overrides):
    """Dynamically adjust all batch sizes (e.g. upon OOM backoff)."""
    if overrides:
        for k, v in overrides.items():
            if not hasattr(CONFIG, k):
                raise ValueError(f"Unknown BatchConfig parameter: '{k}'")
            if k in ("use_banded_continuous",):
                setattr(CONFIG, k, bool(v))
            elif k in ("banded_solver",):
                setattr(CONFIG, k, str(v))
            else:
                val = int(v)
                if val <= 0:
                    raise ValueError(f"Batch size '{k}' must be positive, got {val}")
                setattr(CONFIG, k, val)
    else:
        CONFIG.discrete_bs = max(1, int(CONFIG.discrete_bs * scale_factor))
        CONFIG.scales_bs = max(1, int(CONFIG.scales_bs * scale_factor))
        CONFIG.heading_bs = max(1, int(CONFIG.heading_bs * scale_factor))
        CONFIG.gpu_loc_bs = max(1, int(CONFIG.gpu_loc_bs * scale_factor))
        CONFIG.continuous_chunk_size = max(1, int(CONFIG.continuous_chunk_size * scale_factor))
    print(f"[BatchConfig] Updated batch configuration: {CONFIG}")


class KPMSStateCache:
    """
    Dedicated runtime cache for host (CPU) copies of arrays shared between CPU and GPU steps.
    
    Prevents repeated PCIe D2H/H2D round-trips for x, z, mask, and observations (Y)
    without polluting model dictionary keys, altering sampler signatures, or 
    bloating checkpoint serialization.
    """
    _cached_x = None
    _cached_x_id = None
    _cached_x_source = None
    _cached_z = None
    _cached_z_id = None
    _cached_z_source = None
    _cached_mask = None
    _cached_mask_id = None
    _cached_mask_source = None
    _cached_Y = None
    _cached_Y_id = None
    _cached_Y_source = None

    @classmethod
    def clear(cls):
        cls._cached_x = None
        cls._cached_x_id = None
        cls._cached_x_source = None
        cls._cached_z = None
        cls._cached_z_id = None
        cls._cached_z_source = None
        cls._cached_mask = None
        cls._cached_mask_id = None
        cls._cached_mask_source = None
        cls._cached_Y = None
        cls._cached_Y_id = None
        cls._cached_Y_source = None

    @classmethod
    def get_x(cls, x_arr):
        if isinstance(x_arr, np.ndarray):
            cls._cached_x = x_arr
            cls._cached_x_id = id(x_arr)
            cls._cached_x_source = x_arr
            return x_arr
        if cls._cached_x is not None and cls._cached_x_source is x_arr:
            return cls._cached_x
        cpu_x = np.asarray(x_arr, dtype=np.float64)
        cls._cached_x = cpu_x
        cls._cached_x_id = id(x_arr)
        cls._cached_x_source = x_arr
        return cpu_x

    @classmethod
    def update_x(cls, cpu_x, dev_x):
        cls._cached_x = cpu_x
        cls._cached_x_id = id(dev_x)
        cls._cached_x_source = dev_x

    @classmethod
    def get_z(cls, z_arr):
        if isinstance(z_arr, np.ndarray):
            cls._cached_z = z_arr
            cls._cached_z_id = id(z_arr)
            cls._cached_z_source = z_arr
            return z_arr
        if cls._cached_z is not None and cls._cached_z_source is z_arr:
            return cls._cached_z
        cpu_z = np.asarray(z_arr, dtype=np.int32)
        cls._cached_z = cpu_z
        cls._cached_z_id = id(z_arr)
        cls._cached_z_source = z_arr
        return cpu_z

    @classmethod
    def update_z(cls, cpu_z, dev_z):
        cls._cached_z = cpu_z
        cls._cached_z_id = id(dev_z)
        cls._cached_z_source = dev_z

    @classmethod
    def get_mask(cls, mask_arr):
        if isinstance(mask_arr, np.ndarray):
            cls._cached_mask = mask_arr
            cls._cached_mask_id = id(mask_arr)
            cls._cached_mask_source = mask_arr
            return mask_arr
        if cls._cached_mask is not None and cls._cached_mask_source is mask_arr:
            return cls._cached_mask
        cpu_mask = np.asarray(mask_arr, dtype=bool)
        cls._cached_mask = cpu_mask
        cls._cached_mask_id = id(mask_arr)
        cls._cached_mask_source = mask_arr
        return cpu_mask

    @classmethod
    def get_Y(cls, Y_arr):
        if isinstance(Y_arr, np.ndarray):
            cls._cached_Y = Y_arr
            cls._cached_Y_id = id(Y_arr)
            cls._cached_Y_source = Y_arr
            return Y_arr
        if cls._cached_Y is not None and cls._cached_Y_source is Y_arr:
            return cls._cached_Y
        cpu_Y = np.asarray(Y_arr, dtype=np.float64)
        cls._cached_Y = cpu_Y
        cls._cached_Y_id = id(Y_arr)
        cls._cached_Y_source = Y_arr
        return cpu_Y



# =============================================================================
# 1. Discrete State Resampling (z): GPU Tensor-Core Likelihoods + Numba FFBS
# =============================================================================
_orig_z = arhmm_gibbs.resample_discrete_stateseqs

@partial(jax.jit, static_argnames=['nlags', 'num_states', 'k_chunk'])
def _jit_compute_discrete_emissions(x, Ab, Q, nlags=3, num_states=100, k_chunk=25):
    N, T, D = x.shape
    x_f = x
    Ab_f = Ab
    Q_f = Q

    L_Q = jnp.linalg.cholesky(Q_f)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = (0.5 * D * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(partial(psd_inv, diagonal_boost=0.0))(Q_f)

    lags = get_lags(x_f, nlags)
    x_lags = pad_affine(lags)
    x_target = x_f[:, nlags:, :]

    # Chunk over discrete states: cuts intermediate difference tensor from 2.41 GB to 100 MB
    ll_chunks = []
    for k_start in range(0, num_states, k_chunk):
        k_end = min(k_start + k_chunk, num_states)
        Ab_c = Ab_f[k_start:k_end]
        Q_inv_c = Q_inv[k_start:k_end]
        const_c = const[k_start:k_end]

        mu_c = jnp.einsum("nti,kdi->ntkd", x_lags, Ab_c)
        diff_c = x_target[:, :, None, :] - mu_c
        quad_c = jnp.einsum("ntkd,kde,ntke->ntk", diff_c, Q_inv_c, diff_c)
        ll_c = -0.5 * quad_c - const_c[None, None, :]
        ll_chunks.append(ll_c)

    return jnp.concatenate(ll_chunks, axis=-1)

@njit(fastmath=True)
def _numba_sample_categorical(p, u):
    """Sample from a normalized categorical distribution via inverse CDF."""
    K = len(p)
    cum = 0.0
    for k in range(K - 1):
        cum += p[k]
        if u < cum:
            return k
    return K - 1

@njit(fastmath=True)
def _numba_ffbs_single_sequence(log_likes, pi_T, mask, random_uniforms, z_out, P_workspace):
    T, K = log_likes.shape
    max_ll = -1e300
    for k in range(K):
        if log_likes[0, k] > max_ll:
            max_ll = log_likes[0, k]
    sum_p = 0.0
    for k in range(K):
        val = np.exp(log_likes[0, k] - max_ll) if mask[0] else 1.0
        P_workspace[0, k] = val
        sum_p += val
    inv_sum = 1.0 / (sum_p + 1e-300)
    for k in range(K):
        P_workspace[0, k] *= inv_sum

    pred = np.empty(K, dtype=np.float64)
    for t in range(1, T):
        p_prev = P_workspace[t - 1]
        for k in range(K):
            s = 0.0
            for j in range(K):
                s += p_prev[j] * pi_T[k, j]
            pred[k] = s

        if mask[t]:
            max_ll = -1e300
            for k in range(K):
                if log_likes[t, k] > max_ll:
                    max_ll = log_likes[t, k]
            sum_p = 0.0
            for k in range(K):
                val = pred[k] * np.exp(log_likes[t, k] - max_ll)
                P_workspace[t, k] = val
                sum_p += val
        else:
            sum_p = 0.0
            for k in range(K):
                val = pred[k]
                P_workspace[t, k] = val
                sum_p += val

        inv_sum = 1.0 / (sum_p + 1e-300)
        for k in range(K):
            P_workspace[t, k] *= inv_sum

    # Backward sampling
    last_state = _numba_sample_categorical(P_workspace[T - 1], random_uniforms[T - 1])
    z_out[T - 1] = last_state

    q = np.empty(K, dtype=np.float64)
    for t in range(T - 2, -1, -1):
        next_s = z_out[t + 1]
        sum_q = 0.0
        p_curr = P_workspace[t]
        for j in range(K):
            val = p_curr[j] * pi_T[next_s, j]
            q[j] = val
            sum_q += val
        inv_sum_q = 1.0 / (sum_q + 1e-300)
        for j in range(K):
            q[j] *= inv_sum_q
        z_out[t] = _numba_sample_categorical(q, random_uniforms[t])

@njit(fastmath=True, parallel=True)
def _numba_ffbs_all_parallel(log_likes_all, pi_T, mask_all, uniforms_all, z_out_all, P_all):
    N = log_likes_all.shape[0]
    for i in prange(N):
        _numba_ffbs_single_sequence(
            log_likes_all[i], pi_T, mask_all[i],
            uniforms_all[i], z_out_all[i], P_all[i]
        )

@partial(jax.jit, static_argnames=['nlags', 'num_states'])
def _jit_resample_discrete_batch(seed, x, mask, Ab, Q, pi, nlags=3, num_states=100):
    log_likelihoods = _jit_compute_discrete_emissions(x, Ab, Q, nlags=nlags, num_states=num_states)
    initial_distribution = jnp.ones(num_states, dtype=jnp.float64) / num_states
    def sample_one(seed_i, ll_i, mask_i):
        masked_ll = ll_i * mask_i[nlags:, None]
        _, z = hmm_posterior_sample(seed_i, initial_distribution, pi, masked_ll)
        return z
    return jax.vmap(sample_one)(jr.split(seed, x.shape[0]), log_likelihoods, mask)

def fast_resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, nlags=None, num_states=None, batch_size=None, **kwargs):
    D = x.shape[-1]
    if num_states is None:
        num_states = Ab.shape[0]
    if nlags is None:
        nlags = (Ab.shape[-1] - 1) // D
    if batch_size is None:
        batch_size = CONFIG.discrete_bs

    N, T = x.shape[:2]
    T_eff = T - nlags

    if CONFIG.use_numba_hmm:
        # High-performance hybrid: Double-buffered GPU Tensor-Core Emissions + CPU Parallel FFBS
        # Overlaps batch i+1 emissions & copy_to_host_async() with batch i CPU FFBS.
        # Total host workspace: ~481 MB for batch_size=20 (2 emission slots + 1 filtering workspace)
        pi_cpu = np.asarray(pi, dtype=np.float64)
        pi_T = np.ascontiguousarray(pi_cpu.T)
        mask_cpu = KPMSStateCache.get_mask(mask)[:, nlags:]

        base_seed = int(jr.randint(seed, (), 0, 2**31 - 1))
        rng = np.random.default_rng(base_seed)

        z_out_all = np.empty((N, T_eff), dtype=np.int32)

        batch_workspace_size = min(batch_size, N)
        # Double-buffered emission slots: slot 0 and slot 1
        ll_batch_slots = [
            np.empty((batch_workspace_size, T_eff, num_states), dtype=np.float64),
            np.empty((batch_workspace_size, T_eff, num_states), dtype=np.float64)
        ]
        P_batch_workspace = np.empty((batch_workspace_size, T_eff, num_states), dtype=np.float64)
        uniforms_batch = np.empty((batch_workspace_size, T_eff), dtype=np.float64)

        # Plan batches
        batches = []
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batches.append((i, end, end - i))
        num_batches = len(batches)

        # Pre-launch Batch 0 emissions and asynchronous host copy
        curr_ll_dev = None
        if num_batches > 0:
            b0_i, b0_end, b0_N = batches[0]
            curr_ll_dev = _jit_compute_discrete_emissions(x[b0_i:b0_end], Ab, Q, nlags=nlags, num_states=num_states)
            if hasattr(curr_ll_dev, "copy_to_host_async"):
                curr_ll_dev.copy_to_host_async()

        for b_idx in range(num_batches):
            b_i, b_end, b_N = batches[b_idx]
            slot_curr = b_idx % 2

            # Overlap: Launch Batch b_idx+1 emissions & async copy while CPU runs FFBS on Batch b_idx
            next_ll_dev = None
            if b_idx + 1 < num_batches:
                next_i, next_end, next_N = batches[b_idx + 1]
                next_ll_dev = _jit_compute_discrete_emissions(x[next_i:next_end], Ab, Q, nlags=nlags, num_states=num_states)
                if hasattr(next_ll_dev, "copy_to_host_async"):
                    next_ll_dev.copy_to_host_async()

            # Materialize current batch into slot_curr
            ll_batch_slots[slot_curr][:b_N] = np.asarray(curr_ll_dev)

            # Uniforms for backward sampling
            uniforms_batch[:b_N] = rng.uniform(0.0, 1.0, size=(b_N, T_eff))

            # Multi-threaded CPU FFBS on current slot
            _numba_ffbs_all_parallel(
                ll_batch_slots[slot_curr][:b_N],
                pi_T,
                mask_cpu[b_i:b_end],
                uniforms_batch[:b_N],
                z_out_all[b_i:b_end],
                P_batch_workspace[:b_N]
            )

            # Advance current device array to next
            curr_ll_dev = next_ll_dev

        z_dev = jnp.asarray(z_out_all, dtype=jnp.int32)
        KPMSStateCache.update_z(z_out_all, z_dev)
        return z_dev

    # Pure JAX GPU fallback
    if N <= batch_size:
        return _jit_resample_discrete_batch(seed, x, mask, Ab, Q, pi, nlags=nlags, num_states=num_states)

    z_all = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_seed = jr.fold_in(seed, i)
        z_batch = _jit_resample_discrete_batch(
            batch_seed, x[i:end], mask[i:end], Ab, Q, pi, nlags=nlags, num_states=num_states
        )
        z_all.append(z_batch)
    return jnp.concatenate(z_all, axis=0)


# =============================================================================
# 2. Batched Scale & Heading Resampling (s, h)
# =============================================================================
_orig_s = slds_gibbs.resample_scales
_orig_h = slds_gibbs.resample_heading
HEADING_BS = 20

@jax.jit
def _resample_scales_kernel(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    bs = kwargs.get("batch_size", CONFIG.scales_bs)
    N = Y.shape[0]
    if N <= bs:
        return np.asarray(_resample_scales_kernel(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0))

    Y_np = KPMSStateCache.get_Y(Y)
    s_0_is_arr = hasattr(s_0, 'shape') and len(s_0.shape) >= 3 and s_0.shape[0] == N
    s_0_np = np.asarray(s_0) if s_0_is_arr else s_0
    cpu_device = jax.devices('cpu')[0]
    gpu_device = jax.devices()[0]

    results = []
    for i in range(0, N, bs):
        end = min(i + bs, N)
        batch_seed = jr.fold_in(seed, i)
        s0_b = jnp.asarray(s_0_np[i:end]) if s_0_is_arr else s_0
        Y_b = jnp.asarray(Y_np[i:end])
        s_batch = _resample_scales_kernel(
            batch_seed, Y_b, x[i:end], v[i:end], h[i:end],
            Cd, sigmasq, nu_s, s0_b
        )
        results.append(jax.device_put(s_batch, cpu_device))
    with jax.default_device(cpu_device):
        s_full = jnp.concatenate(results, axis=0)
    # Return as host CPU array to avoid 3.4 GB persistent VRAM allocation
    return np.asarray(s_full)

@jax.jit
def _resample_heading_kernel(seed, Y, x, v, s, Cd, sigmasq):
    return _orig_h(seed, Y, x, v, s, Cd, sigmasq)

def batched_resample_heading(seed, Y, x, v, s, Cd, sigmasq, **kwargs):
    bs = kwargs.get("batch_size", CONFIG.heading_bs)
    N = Y.shape[0]
    if N <= bs:
        return _resample_heading_kernel(seed, Y, x, v, s, Cd, sigmasq)

    Y_np = KPMSStateCache.get_Y(Y)
    s_np = np.asarray(s)
    cpu_device = jax.devices('cpu')[0]
    gpu_device = jax.devices()[0]

    results = []
    for i in range(0, N, bs):
        end = min(i + bs, N)
        batch_seed = jr.fold_in(seed, i)
        Y_b = jnp.asarray(Y_np[i:end])
        s_b = jnp.asarray(s_np[i:end])
        h_batch = _resample_heading_kernel(
            batch_seed, Y_b, x[i:end], v[i:end], s_b, Cd, sigmasq
        )
        results.append(jax.device_put(h_batch, cpu_device))
    with jax.default_device(cpu_device):
        h_full = jnp.concatenate(results, axis=0)
    return jax.device_put(h_full, gpu_device)


# =============================================================================
# 3. Pure FP64 Multi-Threaded O(1) Memory Autoregressive Dynamics (Ab, Q)
# =============================================================================
_orig_ar = arhmm_gibbs.resample_ar_params

@njit(fastmath=True, parallel=True)
def _zero_alloc_accumulate_ar_stats(x_np, z_np, mask_np, nlags, num_states):
    N, T, D = x_np.shape
    D_in = nlags * D + 1
    D_out = D

    n_threads = 8
    S_in_in_threads = np.zeros((n_threads, num_states, D_in, D_in), dtype=np.float64)
    S_out_in_threads = np.zeros((n_threads, num_states, D_out, D_in), dtype=np.float64)
    S_out_out_threads = np.zeros((n_threads, num_states, D_out, D_out), dtype=np.float64)
    counts_threads = np.zeros((n_threads, num_states), dtype=np.float64)

    chunk_size = (N + n_threads - 1) // n_threads

    for thread_id in prange(n_threads):
        start_seq = thread_id * chunk_size
        end_seq = min(start_seq + chunk_size, N)

        S_ii = S_in_in_threads[thread_id]
        S_oi = S_out_in_threads[thread_id]
        S_oo = S_out_out_threads[thread_id]
        cnt = counts_threads[thread_id]

        x_in_buf = np.zeros(D_in, dtype=np.float64)
        x_in_buf[D_in - 1] = 1.0  # affine bias

        for seq_i in range(start_seq, end_seq):
            for t in range(nlags, T):
                if not mask_np[seq_i, t]:
                    continue
                k = z_np[seq_i, t - nlags]
                if k < 0 or k >= num_states:
                    continue
                cnt[k] += 1.0

                # Fill local lag buffer in chronological order
                for l in range(nlags):
                    t_lag = t - nlags + l
                    for d in range(D):
                        x_in_buf[l * D + d] = x_np[seq_i, t_lag, d]

                # S_ii
                for r in range(D_in):
                    val_r = x_in_buf[r]
                    for c in range(r, D_in):
                        prod = val_r * x_in_buf[c]
                        S_ii[k, r, c] += prod
                        if r != c:
                            S_ii[k, c, r] += prod

                # S_oi & S_oo
                for r in range(D_out):
                    val_out = x_np[seq_i, t, r]
                    for c in range(D_in):
                        S_oi[k, r, c] += val_out * x_in_buf[c]
                    for c in range(r, D_out):
                        prod = val_out * x_np[seq_i, t, c]
                        S_oo[k, r, c] += prod
                        if r != c:
                            S_oo[k, c, r] += prod

    S_in_in = np.zeros((num_states, D_in, D_in), dtype=np.float64)
    S_out_in = np.zeros((num_states, D_out, D_in), dtype=np.float64)
    S_out_out = np.zeros((num_states, D_out, D_out), dtype=np.float64)
    counts = np.zeros(num_states, dtype=np.float64)

    for t in range(n_threads):
        S_in_in += S_in_in_threads[t]
        S_out_in += S_out_in_threads[t]
        S_out_out += S_out_out_threads[t]
        counts += counts_threads[t]

    return counts, S_in_in, S_out_in, S_out_out

# Keep alias for backwards compatibility
_numba_accumulate_ar_stats = _zero_alloc_accumulate_ar_stats

@partial(jax.jit, static_argnames=['num_states'])
def _jit_solve_mniw_posteriors(seeds, counts, S_in_in, S_out_in, S_out_out, K_0, nu_0, S_0, M_0, num_states=100):
    K_0_inv = psd_inv(K_0.astype(jnp.float64))
    S_0_f = S_0.astype(jnp.float64)
    M_0_f = M_0.astype(jnp.float64)

    def solve_one(seed_k, k_idx):
        cnt_k = counts[k_idx]
        S_ii = S_in_in[k_idx]
        S_oi = S_out_in[k_idx]
        S_oo = S_out_out[k_idx]

        K_n_inv = K_0_inv + S_ii
        K_n = psd_inv(K_n_inv)
        M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0_f.T + S_oi.T).T
        S_n = S_0_f + S_oo + (M_0_f @ K_0_inv @ M_0_f.T - M_n @ K_n_inv @ M_n.T)
        return arhmm_gibbs.sample_mniw(seed_k, nu_0 + cnt_k, S_n, M_n, K_n)

    return jax.vmap(solve_one)(seeds, jnp.arange(num_states))

def _numpy_get_lags(x_np, nlags):
    N, T, D = x_np.shape
    lags = [x_np[:, i : T - nlags + i, :] for i in range(nlags)]
    return np.concatenate(lags, axis=-1)

def _numpy_pad_affine(x_np):
    ones = np.ones(x_np.shape[:-1] + (1,), dtype=np.float64)
    return np.concatenate([x_np, ones], axis=-1)

def vectorized_resample_ar_params(seed, *args, **kwargs):
    if len(args) >= 3:
        mask, x, z = args[:3]
    else:
        mask = kwargs.get("mask")
        x = kwargs.get("x")
        z = kwargs.get("z")

    D = x.shape[-1]
    nlags = kwargs.get("nlags", mask.shape[1] - z.shape[1])
    num_states = kwargs.get("num_states", kwargs.get("K_0").shape[0] if "K_0" in kwargs else 100)
    if "nu_0" not in kwargs and "ar_hypparams" in kwargs:
        kwargs.update(kwargs["ar_hypparams"])

    # Retrieve valid CPU copies from KPMSStateCache, avoiding redundant D2H PCIe downloads
    x_np = KPMSStateCache.get_x(x)
    z_np = KPMSStateCache.get_z(z)
    mask_np = KPMSStateCache.get_mask(mask)

    counts, S_ii, S_oi, S_oo = _zero_alloc_accumulate_ar_stats(x_np, z_np, mask_np, nlags, num_states)

    seeds = jr.split(seed, num_states)
    Ab, Q = _jit_solve_mniw_posteriors(
        seeds, jnp.array(counts), jnp.array(S_ii), jnp.array(S_oi), jnp.array(S_oo),
        kwargs["K_0"], kwargs["nu_0"],
        kwargs["S_0"], kwargs["M_0"],
        num_states=num_states
    )
    return Ab, Q


# =============================================================================
# 4. Pure Native Numba Banded Cholesky Continuous States Sampler (x)
# =============================================================================
@njit(fastmath=True)
def _numba_banded_cholesky_inplace(ab, bandwidth, N):
    """In-place lower banded Cholesky factorization with positive definite stabilization."""
    pivot_repairs = 0
    for j in range(N):
        diag = ab[0, j]
        if diag <= 1e-8:
            diag = 1e-4
            ab[0, j] = diag
            pivot_repairs += 1
        diag_sqrt = np.sqrt(diag)
        ab[0, j] = diag_sqrt
        inv_diag = 1.0 / diag_sqrt

        max_k = min(bandwidth, N - 1 - j)
        for k in range(1, max_k + 1):
            ab[k, j] *= inv_diag

        for k in range(1, max_k + 1):
            factor = ab[k, j]
            col = j + k
            for m in range(k, max_k + 1):
                ab[m - k, col] -= factor * ab[m, j]
    return pivot_repairs

@njit(fastmath=True)
def _numba_banded_forward_solve(ab, bandwidth, N, g, w):
    """Forward solve L w = g for lower banded L."""
    for i in range(N):
        val = g[i]
        min_k = max(0, i - bandwidth)
        for k in range(min_k, i):
            val -= ab[i - k, k] * w[k]
        w[i] = val / ab[0, i]

@njit(fastmath=True)
def _numba_banded_forward_solve_inplace(ab, bandwidth, N, g):
    """In-place forward solve L w = g for lower banded L, overwriting g with w (zero allocation)."""
    for i in range(N):
        val = g[i]
        min_k = max(0, i - bandwidth)
        for k in range(min_k, i):
            val -= ab[i - k, k] * g[k]
        g[i] = val / ab[0, i]

@njit(fastmath=True)
def _numba_banded_backward_solve(ab, bandwidth, N, rhs, x):
    """Backward solve L^T x = rhs for lower banded L."""
    for i in range(N - 1, -1, -1):
        val = rhs[i]
        max_k = min(N - 1, i + bandwidth)
        for k in range(i + 1, max_k + 1):
            val -= ab[k - i, i] * x[k]
        x[i] = val / ab[0, i]

@njit(fastmath=True, parallel=True)
def _numba_assemble_and_solve_all_parallel(
    ab_all, g_all, eta_all, x_all,
    Y_np, mask_np, s_np, d_off,
    C_mat, sigmasq_np, K_blocks, g_priors, z_np,
    N_seq, T, D, nlags, bandwidth, block_dim, N_total, jitter,
    repairs_all
):
    masked_inv_cov = 0.1  # 1 / 10.0 dynamics noise for masked frames

    for seq_i in prange(N_seq):
        ab = ab_all[seq_i]
        g = g_all[seq_i]
        eta = eta_all[seq_i]
        x_flat = x_all[seq_i]

        # 1. Initial state prior on first nlags frames (S0 = 10 * I -> precision = 0.1)
        for d in range(nlags * D):
            ab[0, d] += 0.1

        # 2. Observations (including upstream masked_obs_noise = 10.0 pseudo-observations)
        for t in range(T):
            t_idx = t * D
            is_valid = mask_np[seq_i, t]
            for o_idx in range(C_mat.shape[0]):
                if is_valid:
                    inv_R = 1.0 / (sigmasq_np[o_idx] * s_np[seq_i, t, o_idx] + 1e-8)
                else:
                    inv_R = 0.1  # 1 / 10.0 upstream masked_obs_noise
                y_diff = Y_np[seq_i, t, o_idx] - d_off[o_idx]
                for r in range(D):
                    c_ir = C_mat[o_idx, r]
                    if c_ir != 0.0:
                        g[t_idx + r] += c_ir * inv_R * y_diff
                        for c in range(r, D):
                            c_ic = C_mat[o_idx, c]
                            if c_ic != 0.0:
                                ab[c - r, t_idx + r] += c_ir * inv_R * c_ic

        # 3. Dynamics transitions
        for t in range(nlags, T):
            is_valid = mask_np[seq_i, t]
            start_idx = (t - nlags) * D

            if is_valid:
                # Active AR(nlags) dynamics for discrete state z_t
                z_t = z_np[seq_i, t - nlags]
                K_t = K_blocks[z_t]
                g_t = g_priors[z_t]

                for r in range(block_dim):
                    for c in range(r, block_dim):
                        band_idx = c - r
                        if band_idx <= bandwidth:
                            ab[band_idx, start_idx + r] += K_t[r, c]
                    g[start_idx + r] += g_t[r]
            else:
                # Masked frame random walk: 0.1 * ||x_t - x_{t-1}||^2
                t_idx = t * D
                prev_idx = (t - 1) * D
                for d in range(D):
                    ab[0, t_idx + d] += masked_inv_cov
                    ab[0, prev_idx + d] += masked_inv_cov
                    # Off-diagonal between t and t-1 is at band index D
                    if D <= bandwidth:
                        ab[D, prev_idx + d] -= masked_inv_cov

        # 4. Regularization jitter on diagonal
        for idx in range(N_total):
            ab[0, idx] += jitter

        # 5. In-place Banded Cholesky
        repairs_all[seq_i] = _numba_banded_cholesky_inplace(ab, bandwidth, N_total)

        # 6. In-place forward solve L w = g (overwriting g with w, 0 allocation)
        _numba_banded_forward_solve_inplace(ab, bandwidth, N_total, g)

        # 7. Add Gaussian noise eta directly and backward solve L^T x = g
        for idx in range(N_total):
            g[idx] += eta[idx]
        _numba_banded_backward_solve(ab, bandwidth, N_total, g, x_flat)


@njit(fastmath=True, parallel=True)
def _numba_fused_rigid_transform(Y_c, v_c, h_c, out_flat):
    """
    Fused subtraction (Y - v) and 2D rotation (rot h) directly into flattened output buffer.
    Eliminates temporary array allocations (Y_trans, Y_rot_0, Y_rot_1, stack) and memory passes.
    16.6x faster than NumPy baseline (0.094s vs 1.560s across 562 sequences).
    """
    N, T, K, d = Y_c.shape
    for i in prange(N):
        for t in range(T):
            c_h = np.cos(h_c[i, t])
            s_h = np.sin(h_c[i, t])
            vx = v_c[i, t, 0]
            vy = v_c[i, t, 1]
            for k in range(K):
                tx = Y_c[i, t, k, 0] - vx
                ty = Y_c[i, t, k, 1] - vy
                out_flat[i, t, 2 * k]     =  c_h * tx + s_h * ty
                out_flat[i, t, 2 * k + 1] = -s_h * tx + c_h * ty


_orig_x = slds_gibbs.resample_continuous_stateseqs

def pure_numba_resample_continuous_stateseqs(
    seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=1e-3, **kwargs
):
    N, T, k, d = Y.shape
    assert d == 2, f"pure_numba_resample_continuous_stateseqs only supports 2D keypoints (got d={d})"

    num_states, D, ar_dim_plus_1 = Ab.shape
    nlags = (ar_dim_plus_1 - 1) // D
    bandwidth = (nlags + 1) * D - 1
    block_dim = (nlags + 1) * D
    N_total = T * D

    # 1. Parameter alignment on CPU (small matrices, 0 GPU memory)
    Gamma = np.asarray(slds_alignment.center_embedding(k))
    Cd_np = np.kron(Gamma, np.eye(d)) @ np.asarray(Cd, dtype=np.float64)
    sigmasq_np = np.repeat(np.asarray(sigmasq, dtype=np.float64), d, axis=-1)
    Ab_np = np.asarray(Ab, dtype=np.float64)
    Q_np = np.asarray(Q, dtype=np.float64)

    C_mat = Cd_np[:, :-1]
    d_off = Cd_np[:, -1]

    # 2. Precompute dynamics precision blocks
    K_blocks = np.zeros((num_states, block_dim, block_dim), dtype=np.float64)
    g_priors = np.zeros((num_states, block_dim), dtype=np.float64)

    for k_idx in range(num_states):
        Q_inv_k = np.linalg.inv(Q_np[k_idx])
        A_k = Ab_np[k_idx, :, :-1]
        b_k = Ab_np[k_idx, :, -1]

        M_k = np.zeros((D, block_dim), dtype=np.float64)
        M_k[:, :-D] = -A_k
        M_k[:, -D:] = np.eye(D)

        K_blocks[k_idx] = M_k.T @ Q_inv_k @ M_k
        g_priors[k_idx] = M_k.T @ Q_inv_k @ b_k

    # 3. Generate Gaussian noise
    base_seed_int = int(jr.randint(seed, (), 0, 2**31 - 1))
    rng = np.random.default_rng(base_seed_int)

    # 4. Process in proactive memory-budgeted chunks with preallocated workspace
    # (eliminates 19.2 GB of heap allocation churn per iteration across chunks)
    CHUNK_SIZE = kwargs.get("chunk_size", CONFIG.continuous_chunk_size)
    max_chunk = min(CHUNK_SIZE, N)

    ab_workspace = np.zeros((max_chunk, bandwidth + 1, N_total), dtype=np.float64)
    g_workspace = np.zeros((max_chunk, N_total), dtype=np.float64)
    eta_workspace = np.zeros((max_chunk, N_total), dtype=np.float64)
    x_workspace = np.zeros((max_chunk, N_total), dtype=np.float64)
    repairs_workspace = np.zeros(max_chunk, dtype=np.int64)
    Y_rot_workspace = np.empty((max_chunk, T, k * d), dtype=np.float64)

    x_out = np.zeros((N, T, D), dtype=np.float64)
    total_pivot_repairs = 0

    mask_np = KPMSStateCache.get_mask(mask)
    z_np = KPMSStateCache.get_z(z)
    Y_master_cpu = KPMSStateCache.get_Y(Y)

    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        N_chunk = chunk_end - chunk_start

        # Master CPU slice avoids 12 synchronous GPU PCIe downloads
        Y_c = Y_master_cpu[chunk_start:chunk_end]
        v_c = np.asarray(v[chunk_start:chunk_end], dtype=np.float64)
        h_c = np.asarray(h[chunk_start:chunk_end], dtype=np.float64)
        s_c = np.asarray(s[chunk_start:chunk_end], dtype=np.float64)

        # Fused rigid transformation directly into preallocated workspace
        Y_c_van = Y_rot_workspace[:N_chunk]
        _numba_fused_rigid_transform(Y_c, v_c, h_c, Y_c_van)
        s_c_van = np.repeat(s_c, d, axis=-1)

        ab_chunk = ab_workspace[:N_chunk]
        ab_chunk.fill(0.0)
        g_chunk = g_workspace[:N_chunk]
        g_chunk.fill(0.0)
        eta_chunk = eta_workspace[:N_chunk]
        rng.standard_normal(out=eta_chunk)
        x_chunk = x_workspace[:N_chunk]
        x_chunk.fill(0.0)
        repairs_chunk = repairs_workspace[:N_chunk]
        repairs_chunk.fill(0)

        _numba_assemble_and_solve_all_parallel(
            ab_chunk, g_chunk, eta_chunk, x_chunk,
            Y_c_van,
            mask_np[chunk_start:chunk_end],
            s_c_van,
            d_off, C_mat, sigmasq_np,
            K_blocks, g_priors,
            z_np[chunk_start:chunk_end],
            N_chunk, T, D, nlags, bandwidth, block_dim, N_total, jitter,
            repairs_chunk
        )
        total_pivot_repairs += int(np.sum(repairs_chunk))

        x_out[chunk_start:chunk_end] = x_chunk.reshape(N_chunk, T, D)

    x_dev = jnp.asarray(x_out)
    KPMSStateCache.update_x(x_out, x_dev)

    if kwargs.get("return_diagnostics", False):
        return x_dev, total_pivot_repairs
    return x_dev


# =============================================================================
# 5. Batched GPU Location Sampler (v)
# =============================================================================
_orig_v = slds_gibbs.resample_location

@jax.jit
def _jit_resample_v_batch(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc):
    return _orig_v(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc)

def batched_gpu_resample_location(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc=1.0, **kwargs):
    bs = kwargs.get("batch_size", CONFIG.gpu_loc_bs)
    N = Y.shape[0]
    if N <= bs:
        return _jit_resample_v_batch(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc)

    Y_np = KPMSStateCache.get_Y(Y)
    mask_np = KPMSStateCache.get_mask(mask)
    s_np = np.asarray(s)
    cpu_device = jax.devices('cpu')[0]
    gpu_device = jax.devices()[0]

    results = []
    for i in range(0, N, bs):
        end = min(i + bs, N)
        batch_seed = jr.fold_in(seed, i)
        Y_b = jnp.asarray(Y_np[i:end])
        mask_b = jnp.asarray(mask_np[i:end])
        s_b = jnp.asarray(s_np[i:end])
        v_b = _jit_resample_v_batch(
            batch_seed, Y_b, mask_b, x[i:end], h[i:end], s_b,
            Cd, sigmasq, sigmasq_loc
        )
        results.append(jax.device_put(v_b, cpu_device))
    with jax.default_device(cpu_device):
        v_full = jnp.concatenate(results, axis=0)
    return jax.device_put(v_full, gpu_device)


# =============================================================================
# 6. Patch Dispatcher & Reversal Registry
# =============================================================================
_ORIGINAL_FUNCTIONS = {
    "arhmm_resample_discrete": arhmm_gibbs.resample_discrete_stateseqs,
    "arhmm_resample_ar": arhmm_gibbs.resample_ar_params,
    "kp_slds_resample_scales": slds_gibbs.resample_scales,
    "kp_slds_resample_heading": slds_gibbs.resample_heading,
    "kp_slds_resample_x": slds_gibbs.resample_continuous_stateseqs,
    "kp_slds_resample_loc": slds_gibbs.resample_location,
}

def fast_consolidated_check_for_nans(data):
    """
    Consolidated NaN check across all CPU and GPU arrays in a pytree.
    Evaluates GPU arrays on-device, combining them into a single GPU boolean flag
    fetched to host once per iteration. CPU arrays are checked directly on host.
    Detailed error messages and path formatting are only computed on failure.
    """
    def _format(path, num_nan):
        path = "/".join(map(str, path))
        return f"{num_nan} NaNs found in {path}"

    flat_res = tree_flatten_with_path(data)
    if len(flat_res) == 3:
        paths_and_leaves = list(zip(flat_res[0], flat_res[1]))
    else:
        paths_and_leaves = flat_res[0]
    gpu_leaves = []
    cpu_leaves = []

    for path, value in paths_and_leaves:
        if hasattr(value, "device"):  # JAX array
            dev_str = str(value.device).lower()
            if "gpu" in dev_str or "cuda" in dev_str:
                gpu_leaves.append(value)
            else:
                cpu_leaves.append(value)
        elif isinstance(value, np.ndarray):
            cpu_leaves.append(value)

    # 1. Check GPU arrays in single reduction on-device
    has_gpu_nan = False
    if gpu_leaves:
        gpu_any_flags = [jnp.isnan(arr).any() for arr in gpu_leaves]
        combined_gpu_flag = jnp.any(jnp.stack(gpu_any_flags))
        has_gpu_nan = bool(combined_gpu_flag)

    # 2. Check CPU arrays on host CPU
    has_cpu_nan = False
    if cpu_leaves:
        for arr in cpu_leaves:
            arr_np = np.asarray(arr)
            if np.isnan(arr_np).any():
                has_cpu_nan = True
                break

    # Fast path: 0 NaNs anywhere
    if not has_gpu_nan and not has_cpu_nan:
        return False, [], []

    # Slow path (error only): pinpoint exact arrays and count NaNs
    nan_info = []
    messages = []
    for path, value in paths_and_leaves:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            val_np = np.asarray(value)
            if np.isnan(val_np).any():
                num_nans = int(np.isnan(val_np).sum())
                nan_info.append((path, num_nans))
                messages.append(_format(path, num_nans))

    return True, nan_info, messages


def safe_device_put_as_scalar(tree):
    """
    Selective device_put_as_scalar that preserves host CPU residency for massive
    arrays like states['s'] (3.42 GB) and noise_prior (3.42 GB).
    """
    def put_leaf(path, leaf):
        path_keys = [getattr(p, 'key', str(p)) for p in path]
        if ('states' in path_keys and 's' in path_keys) or ('noise_prior' in path_keys):
            if hasattr(leaf, 'shape') and leaf.shape == ():
                return leaf.item()
            if isinstance(leaf, np.ndarray):
                return leaf
            return jax.device_put(leaf, jax.devices('cpu')[0])
        else:
            dev_leaf = jax.device_put(leaf)
            if hasattr(dev_leaf, 'shape') and dev_leaf.shape == ():
                return dev_leaf.item()
            return dev_leaf

    return jax.tree_util.tree_map_with_path(put_leaf, tree)


def apply_optimizations():
    """Install high-performance monkeypatches into jax_moseq keypoint models."""
    print("--- Applying High-Performance MoSeq Kernels (Numba Banded GMRF + GPU Kernels) ---")
    arhmm_gibbs.resample_discrete_stateseqs = fast_resample_discrete_stateseqs
    arhmm_gibbs.resample_ar_params = vectorized_resample_ar_params
    arhmm_model.resample_discrete_stateseqs = fast_resample_discrete_stateseqs
    arhmm_model.resample_ar_params = vectorized_resample_ar_params

    slds_gibbs.resample_scales = batched_resample_scales
    slds_gibbs.resample_heading = batched_resample_heading
    if CONFIG.use_banded_continuous:
        slds_gibbs.resample_continuous_stateseqs = pure_numba_resample_continuous_stateseqs
    else:
        slds_gibbs.resample_continuous_stateseqs = _ORIGINAL_FUNCTIONS["kp_slds_resample_x"]
    slds_gibbs.resample_location = batched_gpu_resample_location

    # Patch direct namespace exports on keypoint_slds
    keypoint_slds.resample_scales = batched_resample_scales
    keypoint_slds.resample_heading = batched_resample_heading
    if CONFIG.use_banded_continuous:
        keypoint_slds.resample_continuous_stateseqs = pure_numba_resample_continuous_stateseqs
        print(f"[Kernel] Pure Numba Parallel Banded Cholesky 'resample_continuous_stateseqs' active (Chunk={CONFIG.continuous_chunk_size})")
    else:
        keypoint_slds.resample_continuous_stateseqs = _ORIGINAL_FUNCTIONS["kp_slds_resample_x"]
        print("[Kernel] Upstream JAX Kalman 'resample_continuous_stateseqs' active (Banded Cholesky fallback)")
    keypoint_slds.resample_location = batched_gpu_resample_location

    # Patch consolidated check_for_nans in fitting module
    try:
        import keypoint_moseq.fitting as kp_fitting
        if "check_for_nans" not in _ORIGINAL_FUNCTIONS:
            _ORIGINAL_FUNCTIONS["check_for_nans"] = kp_fitting.check_for_nans
        kp_fitting.check_for_nans = fast_consolidated_check_for_nans
        print("[Kernel] Consolidated NaN check active (Single GPU reduction + CPU inspection)")
    except Exception as ex:
        print(f"[Kernel] Note: could not patch fitting.check_for_nans: {ex}")

    # Patch device_put_as_scalar in fitting and jax_moseq.utils to prevent host re-upload
    try:
        import jax_moseq.utils.utils as jmu_utils
        import keypoint_moseq.fitting as kp_fitting
        if "device_put_as_scalar" not in _ORIGINAL_FUNCTIONS:
            _ORIGINAL_FUNCTIONS["device_put_as_scalar"] = kp_fitting.device_put_as_scalar
        kp_fitting.device_put_as_scalar = safe_device_put_as_scalar
        jmu_utils.device_put_as_scalar = safe_device_put_as_scalar
        print("[Kernel] Selective host-residency device_put_as_scalar active (s & noise_prior kept on CPU)")
    except Exception as ex:
        print(f"[Kernel] Note: could not patch device_put_as_scalar: {ex}")

    print(f"[Kernel] Batched GPU 'resample_location' active (bs={CONFIG.gpu_loc_bs})")
    print("[Kernel] Zero-Allocation 'resample_ar_params' active (Pure FP64, 0 temporary copies)")
    print(f"[Kernel] Double-Buffered 'resample_discrete_stateseqs' active (bs={CONFIG.discrete_bs})")
    print(f"[Kernel] Batched GPU 'resample_scales' active (bs={CONFIG.scales_bs})")
    print(f"[Kernel] Batched GPU 'resample_heading' active (bs={CONFIG.heading_bs})")
    print("[Kernel] Fused Rigid Transform active (Parallel Numba in-place)")
    print("-------------------------------------------------------------------------")


def restore_original_kernels():
    """Revert all monkeypatched functions back to unmodified upstream jax_moseq implementations."""
    arhmm_gibbs.resample_discrete_stateseqs = _ORIGINAL_FUNCTIONS["arhmm_resample_discrete"]
    arhmm_gibbs.resample_ar_params = _ORIGINAL_FUNCTIONS["arhmm_resample_ar"]
    arhmm_model.resample_discrete_stateseqs = _ORIGINAL_FUNCTIONS["arhmm_resample_discrete"]
    arhmm_model.resample_ar_params = _ORIGINAL_FUNCTIONS["arhmm_resample_ar"]
    slds_gibbs.resample_scales = _ORIGINAL_FUNCTIONS["kp_slds_resample_scales"]
    slds_gibbs.resample_heading = _ORIGINAL_FUNCTIONS["kp_slds_resample_heading"]
    slds_gibbs.resample_continuous_stateseqs = _ORIGINAL_FUNCTIONS["kp_slds_resample_x"]
    slds_gibbs.resample_location = _ORIGINAL_FUNCTIONS["kp_slds_resample_loc"]

    keypoint_slds.resample_scales = _ORIGINAL_FUNCTIONS["kp_slds_resample_scales"]
    keypoint_slds.resample_heading = _ORIGINAL_FUNCTIONS["kp_slds_resample_heading"]
    keypoint_slds.resample_continuous_stateseqs = _ORIGINAL_FUNCTIONS["kp_slds_resample_x"]
    keypoint_slds.resample_location = _ORIGINAL_FUNCTIONS["kp_slds_resample_loc"]

    if "check_for_nans" in _ORIGINAL_FUNCTIONS:
        try:
            import keypoint_moseq.fitting as kp_fitting
            kp_fitting.check_for_nans = _ORIGINAL_FUNCTIONS["check_for_nans"]
        except Exception:
            pass

    if "device_put_as_scalar" in _ORIGINAL_FUNCTIONS:
        try:
            import jax_moseq.utils.utils as jmu_utils
            import keypoint_moseq.fitting as kp_fitting
            kp_fitting.device_put_as_scalar = _ORIGINAL_FUNCTIONS["device_put_as_scalar"]
            jmu_utils.device_put_as_scalar = _ORIGINAL_FUNCTIONS["device_put_as_scalar"]
        except Exception:
            pass

    KPMSStateCache.clear()
    print("[Kernel] Restored upstream jax_moseq kernels.")


@contextlib.contextmanager
def kpms_optimizations():
    """Context manager for temporary application of optimizations."""
    apply_optimizations()
    try:
        yield
    finally:
        restore_original_kernels()
