"""KPMS kernels / autoregression; numerical algorithms preserved from the baseline."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import numpy as np
from jax_moseq.utils import psd_inv, psd_solve
from numba import njit, prange

from fpcode.kpms.backend.state import KPMSStateCache

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
                for lag_i in range(nlags):
                    t_lag = t - nlags + lag_i
                    for d in range(D):
                        x_in_buf[lag_i * D + d] = x_np[seq_i, t_lag, d]


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
