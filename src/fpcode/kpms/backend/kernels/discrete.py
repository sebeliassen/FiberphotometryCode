"""KPMS kernels / discrete; numerical algorithms preserved from the baseline."""

from __future__ import annotations

from functools import partial

import dynamax.hidden_markov_model.inference as hmm_inference
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import numpy as np
from jax_moseq.utils import pad_affine, psd_inv
from jax_moseq.utils.autoregression import get_lags
from numba import njit, prange

from fpcode.kpms.backend.configuration import CONFIG
from fpcode.kpms.backend.state import KPMSStateCache

hmm_posterior_sample = hmm_inference.hmm_posterior_sample


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


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
