"""KPMS kernels / location; numerical algorithms preserved from the baseline."""

from __future__ import annotations

from fpcode.kpms.backend.configuration import CONFIG
from fpcode.kpms.backend.state import KPMSStateCache
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import numpy as np


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
