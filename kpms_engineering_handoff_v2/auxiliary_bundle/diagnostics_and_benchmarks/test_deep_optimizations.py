import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)

from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_posterior_sample,
)
from jax_moseq.utils import psd_inv, psd_solve, symmetrize


def benchmark_hmm():
    print("=" * 70)
    print("   1. HMM SAMPLING OPTIMIZATION BENCHMARK (z)")
    print("=" * 70)
    T = 10030
    K = 100
    key = jr.PRNGKey(42)

    init_dist = jnp.ones(K) / K
    pi = jr.dirichlet(key, jnp.ones(K) * 0.1, shape=(K,))
    ll = jr.normal(key, shape=(T, K))

    # Standard Dynamax HMM
    @jax.jit
    def std_hmm(k):
        return hmm_posterior_sample(k, init_dist, pi, ll)

    # Optimized Fused Log-Space Gumbel HMM
    log_A = jnp.log(pi + 1e-12)
    log_init = jnp.log(init_dist + 1e-12)

    @jax.jit
    def opt_hmm(k):
        # 1. Forward pass (log-space scan)
        def fwd_step(carry, t):
            log_pred = carry
            log_filt = log_pred + ll[t]
            log_norm = jax.nn.logsumexp(log_filt)
            log_filt = log_filt - log_norm
            log_pred_next = jax.nn.logsumexp(log_filt[:, None] + log_A, axis=0)
            return log_pred_next, log_filt

        _, log_filts = jax.lax.scan(fwd_step, log_init, jnp.arange(T))

        # 2. Parallel Gumbel noise generation for all timesteps
        gumbel_key, _ = jr.split(k)
        gumbels = jr.gumbel(gumbel_key, shape=(T, K))

        # 3. Backward sampling with Gumbel argmax (no sequential jr.choice / splits)
        last_z = jnp.argmax(log_filts[-1] + gumbels[-1])

        def bwd_step(next_z, t):
            scores = log_filts[t] + log_A[:, next_z] + gumbels[t]
            curr_z = jnp.argmax(scores)
            return curr_z, curr_z

        _, z_seq = jax.lax.scan(bwd_step, last_z, jnp.arange(T - 2, -1, -1))
        return jnp.concatenate([z_seq[::-1], last_z[None]])

    # Warmup
    _ = std_hmm(key)[1].block_until_ready()
    _ = opt_hmm(key).block_until_ready()

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = std_hmm(key)[1].block_until_ready()
    t_std = ((time.perf_counter() - t0) / iters) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = opt_hmm(key).block_until_ready()
    t_opt = ((time.perf_counter() - t0) / iters) * 1000.0

    print(f"Standard Dynamax HMM (z)     : {t_std:7.2f} ms")
    print(f"Optimized Gumbel HMM (z)     : {t_opt:7.2f} ms")
    print(f"Speedup                      : {t_std / t_opt:7.2f}x\n")


def benchmark_kalman():
    print("=" * 70)
    print("   2. KALMAN / LGSSM SAMPLING OPTIMIZATION BENCHMARK (x)")
    print("=" * 70)
    T = 10030
    D_state = 30
    D_obs = 30
    key = jr.PRNGKey(42)

    F = jnp.eye(D_state) * 0.95
    Q = jnp.eye(D_state) * 0.1
    H = jnp.eye(D_obs, D_state)
    R_diag = jnp.ones(D_obs) * 0.2
    y = jr.normal(key, shape=(T, D_obs))

    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=jnp.zeros(D_state), cov=jnp.eye(D_state) * 10.0),
        dynamics=ParamsLGSSMDynamics(weights=F, bias=jnp.zeros(D_state), cov=Q, input_weights=jnp.zeros((D_state, 0))),
        emissions=ParamsLGSSMEmissions(weights=H, bias=jnp.zeros(D_obs), cov=R_diag, input_weights=jnp.zeros((D_obs, 0)))
    )

    # Standard Dynamax Kalman
    @jax.jit
    def std_kalman(k):
        return lgssm_posterior_sample(k, params, y, jitter=1e-3)

    # Lean Kalman (Omits unused marginal log-likelihood computation in forward pass)
    @jax.jit
    def lean_kalman(k):
        # Forward pass without calculating unused log-likelihoods
        def fwd_step(carry, t):
            pred_m, pred_P = carry
            # Condition on emission using Woodbury
            U = H @ jnp.linalg.cholesky(pred_P)
            X = U / R_diag[:, None]
            I_d = jnp.eye(D_state)
            S_inv = jnp.diag(1.0 / R_diag) - X @ psd_solve(I_d + U.T @ X, X.T)
            K = pred_P @ H.T @ S_inv
            S = jnp.diag(R_diag) + H @ pred_P @ H.T
            residual = y[t] - H @ pred_m
            filt_m = pred_m + K @ residual
            filt_P = symmetrize(pred_P - K @ S @ K.T)

            # Predict next
            next_m = F @ filt_m
            next_P = F @ filt_P @ F.T + Q
            return (next_m, next_P), (filt_m, filt_P)

        carry0 = (params.initial.mean, params.initial.cov)
        _, (filt_means, filt_covs) = jax.lax.scan(fwd_step, carry0, jnp.arange(T))

        # Backward sampling
        key_init, key_scan = jr.split(k)
        L_last = jnp.linalg.cholesky(filt_covs[-1] + jnp.eye(D_state) * 1e-3)
        last_x = filt_means[-1] + L_last @ jr.normal(key_init, shape=(D_state,))

        # Pre-generate standard normal noise for all timesteps
        noise_all = jr.normal(key_scan, shape=(T - 1, D_state))

        def bwd_step(next_x, args):
            t, noise = args
            m_f, P_f = filt_means[t], filt_covs[t]
            # Condition on next_x
            pred_cov = F @ P_f @ F.T + Q
            J = P_f @ F.T @ psd_inv(pred_cov)
            smooth_m = m_f + J @ (next_x - F @ m_f)
            smooth_P = symmetrize(P_f - J @ pred_cov @ J.T) + jnp.eye(D_state) * 1e-3
            L = jnp.linalg.cholesky(smooth_P)
            curr_x = smooth_m + L @ noise
            return curr_x, curr_x

        _, x_seq = jax.lax.scan(bwd_step, last_x, (jnp.arange(T - 2, -1, -1), noise_all[::-1]))
        return jnp.vstack([x_seq[::-1], last_x])

    # Warmup
    _ = std_kalman(key).block_until_ready()
    _ = lean_kalman(key).block_until_ready()

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = std_kalman(key).block_until_ready()
    t_std = ((time.perf_counter() - t0) / iters) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = lean_kalman(key).block_until_ready()
    t_lean = ((time.perf_counter() - t0) / iters) * 1000.0

    print(f"Standard Dynamax Kalman (x)  : {t_std:7.2f} ms")
    print(f"Lean Kalman (x)              : {t_lean:7.2f} ms")
    print(f"Speedup                      : {t_std / t_lean:7.2f}x\n")


if __name__ == "__main__":
    benchmark_hmm()
    benchmark_kalman()
