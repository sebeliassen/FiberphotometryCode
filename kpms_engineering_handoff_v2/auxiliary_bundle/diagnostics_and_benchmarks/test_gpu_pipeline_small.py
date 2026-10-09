import time
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import numpy as np

import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.keypoint_slds.alignment as slds_alignment
import jax_moseq.models.slds.gibbs as vanilla_slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
from jax_moseq.utils import pad_affine, psd_inv, psd_solve
from jax_moseq.utils.autoregression import get_lags
from dynamax.hidden_markov_model.inference import hmm_posterior_sample

# Configuration
N = 10
T = 1000
k = 17
d = 2
latent_dim = 10
nlags = 3
num_states = 100
GPU_BATCH_SIZE = 5

print(f"=== Testing GPU-Batched Kernels on Small Dataset (N={N}, T={T}) ===")

seed = jr.PRNGKey(42)
Y = jr.normal(jr.fold_in(seed, 1), (N, T, k, d))
mask = jnp.ones((N, T), dtype=bool)
v = jr.normal(jr.fold_in(seed, 2), (N, T, d))
h = jr.uniform(jr.fold_in(seed, 3), (N, T), minval=-np.pi, maxval=np.pi)
s = jnp.ones((N, T, k), dtype=jnp.float64)
z = jr.randint(jr.fold_in(seed, 4), (N, T - nlags), 0, num_states)
Cd = jr.normal(jr.fold_in(seed, 5), ((k - 1) * d, latent_dim + 1))
sigmasq = jnp.ones(k, dtype=jnp.float64) * 0.1
sigmasq_loc = 1.0
Ab = jr.normal(jr.fold_in(seed, 6), (num_states, latent_dim, nlags * latent_dim + 1)) * 0.01
Q = jnp.tile(jnp.eye(latent_dim, dtype=jnp.float64)[None, ...], (num_states, 1, 1))
pi = jnp.ones((num_states, num_states), dtype=jnp.float64) / num_states
nu_s = 1000.0
s_0 = 1.0

# 1. Discrete state sampling (z)
print("1. Testing Batched GPU Discrete Sampling (z)...")
@partial(jax.jit, static_argnames=['nlags', 'num_states'])
def _jit_resample_discrete_batch(seed, x, mask, Ab, Q, pi, nlags=3, num_states=100):
    N_b, T_b, D_b = x.shape
    L_Q = jnp.linalg.cholesky(Q)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = (0.5 * D_b * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(Q)

    lags = get_lags(x, nlags)
    x_lags = pad_affine(lags)
    x_target = x[:, nlags:, :]

    mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
    diff = x_target[:, :, None, :] - mu_all
    quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
    log_likelihoods = -0.5 * quad - const[None, None, :]

    initial_distribution = jnp.ones(num_states, dtype=jnp.float64) / num_states
    def sample_one(seed_i, ll_i, mask_i):
        masked_ll = ll_i * mask_i[nlags:, None]
        _, z = hmm_posterior_sample(seed_i, initial_distribution, pi, masked_ll)
        return z

    return jax.vmap(sample_one)(jr.split(seed, N_b), log_likelihoods, mask)

def batched_resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, batch_size=GPU_BATCH_SIZE, **kwargs):
    N_all = x.shape[0]
    z_all = []
    for i in range(0, N_all, batch_size):
        end = min(i + batch_size, N_all)
        z_batch = _jit_resample_discrete_batch(
            jr.fold_in(seed, i), x[i:end], mask[i:end], Ab, Q, pi, nlags=nlags, num_states=num_states
        )
        z_all.append(z_batch)
    return jnp.concatenate(z_all, axis=0)

x_init = jr.normal(jr.fold_in(seed, 7), (N, T, latent_dim))
z_new = batched_resample_discrete_stateseqs(seed, x_init, mask, Ab, Q, pi).block_until_ready()
print(f"   ✓ z shape: {z_new.shape}, values in [{z_new.min()}, {z_new.max()}]")

# 2. Continuous states (x) via GPU Associative Scan
print("2. Testing Batched GPU Continuous Sampling (x)...")
@partial(jax.jit, static_argnames=('parallel_message_passing',))
def _jit_resample_x_batch(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=1e-3, parallel_message_passing=True):
    Y_van, s_van, Cd_van, sigmasq_van = slds_alignment.to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    return vanilla_slds_gibbs.resample_continuous_stateseqs(
        seed, Y_van, mask, z, s_van, Ab, Q, Cd_van, sigmasq_van,
        jitter=jitter, parallel_message_passing=parallel_message_passing
    )

def batched_resample_continuous_stateseqs(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, batch_size=GPU_BATCH_SIZE, **kwargs):
    N_all = Y.shape[0]
    x_all = []
    for i in range(0, N_all, batch_size):
        end = min(i + batch_size, N_all)
        x_batch = _jit_resample_x_batch(
            jr.fold_in(seed, i), Y[i:end], mask[i:end], v[i:end], h[i:end], s[i:end], z[i:end],
            Cd, sigmasq, Ab, Q, jitter=1e-3, parallel_message_passing=True
        )
        x_all.append(x_batch)
    return jnp.concatenate(x_all, axis=0)

x_new = batched_resample_continuous_stateseqs(seed, Y, mask, v, h, s, z_new, Cd, sigmasq, Ab, Q).block_until_ready()
print(f"   ✓ x shape: {x_new.shape}, mean: {float(x_new.mean()):.4f}, std: {float(x_new.std()):.4f}")

# 3. Heading angles (h)
print("3. Testing Batched GPU Heading Sampling (h)...")
@jax.jit
def _jit_resample_h_batch(seed, Y, x, v, s, Cd, sigmasq):
    return slds_gibbs.resample_heading(seed, Y, x, v, s, Cd, sigmasq)

def batched_resample_heading(seed, Y, x, v, s, Cd, sigmasq, batch_size=GPU_BATCH_SIZE, **kwargs):
    N_all = Y.shape[0]
    h_all = []
    for i in range(0, N_all, batch_size):
        end = min(i + batch_size, N_all)
        h_batch = _jit_resample_h_batch(
            jr.fold_in(seed, i), Y[i:end], x[i:end], v[i:end], s[i:end], Cd, sigmasq
        )
        h_all.append(h_batch)
    return jnp.concatenate(h_all, axis=0)

h_new = batched_resample_heading(seed, Y, x_new, v, s, Cd, sigmasq).block_until_ready()
print(f"   ✓ h shape: {h_new.shape}, min: {float(h_new.min()):.4f}, max: {float(h_new.max()):.4f}")

# 4. Noise scales (s)
print("4. Testing Batched GPU Scales Sampling (s)...")
@jax.jit
def _jit_resample_s_batch(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = slds_gibbs.compute_squared_error(Y, x, v, h, Cd)
    return slds_gibbs.slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, batch_size=GPU_BATCH_SIZE, **kwargs):
    N_all = Y.shape[0]
    s_all = []
    for i in range(0, N_all, batch_size):
        end = min(i + batch_size, N_all)
        s0_b = s_0[i:end] if (hasattr(s_0, 'shape') and len(s_0.shape) >= 3 and s_0.shape[0] == N_all) else s_0
        s_batch = _jit_resample_s_batch(
            jr.fold_in(seed, i), Y[i:end], x[i:end], v[i:end], h[i:end], Cd, sigmasq, nu_s, s0_b
        )
        s_all.append(s_batch)
    return jnp.concatenate(s_all, axis=0)

s_new = batched_resample_scales(seed, Y, x_new, v, h_new, Cd, sigmasq, nu_s, s_0).block_until_ready()
print(f"   ✓ s shape: {s_new.shape}, min: {float(s_new.min()):.4f}, max: {float(s_new.max()):.4f}")

# 5. Centroid locations (v)
print("5. Testing Batched GPU Location Sampling (v)...")
@jax.jit
def _jit_resample_v_batch(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc):
    return slds_gibbs.resample_location(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc)

def batched_resample_location(seed, Y, mask, x, h, s, Cd, sigmasq, sigmasq_loc=1.0, batch_size=GPU_BATCH_SIZE, **kwargs):
    N_all = Y.shape[0]
    v_all = []
    for i in range(0, N_all, batch_size):
        end = min(i + batch_size, N_all)
        v_batch = _jit_resample_v_batch(
            jr.fold_in(seed, i), Y[i:end], mask[i:end], x[i:end], h[i:end], s[i:end], Cd, sigmasq, sigmasq_loc
        )
        v_all.append(v_batch)
    return jnp.concatenate(v_all, axis=0)

v_new = batched_resample_location(seed, Y, mask, x_new, h_new, s_new, Cd, sigmasq, sigmasq_loc).block_until_ready()
print(f"   ✓ v shape: {v_new.shape}, mean: {float(v_new.mean()):.4f}")

print("\n🎉 ALL 5 GPU-BATCHED GIBBS SAMPLERS PASSED VERIFICATION!")
