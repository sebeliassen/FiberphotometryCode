"""KPMS kernels / scales heading; numerical algorithms preserved from the baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import numpy as np
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error

from fpcode.kpms.backend.configuration import CONFIG
from fpcode.kpms.backend.state import KPMSStateCache

_orig_s = slds_gibbs.resample_scales


_orig_h = slds_gibbs.resample_heading


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
