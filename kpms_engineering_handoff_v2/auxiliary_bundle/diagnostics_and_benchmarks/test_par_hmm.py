import time

import numpy as np

np.bool8 = np.bool_
import jax
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model.inference import hmm_filter, parallel_hmm_filter


def test_parallel_hmm():
    print("Benchmarking Dynamax parallel_hmm_filter vs hmm_filter...")
    jax.config.update("jax_enable_x64", True)

    K = 100
    T = 10030
    N = 2

    seed = jr.PRNGKey(42)
    initial_dist = jnp.ones(K) / K
    pi = jnp.eye(K) * 0.9 + 0.1 / K
    pi = pi / pi.sum(axis=-1, keepdims=True)

    log_likes = jr.normal(seed, (N, T, K))

    @jax.jit
    def seq_filter(lls):
        return jax.vmap(lambda ll: hmm_filter(initial_dist, pi, ll))(lls)

    @jax.jit
    def par_filter(lls):
        return jax.vmap(lambda ll: parallel_hmm_filter(initial_dist, pi, ll))(lls)

    # Warmup
    _ = seq_filter(log_likes)
    _ = par_filter(log_likes)

    t0 = time.perf_counter()
    for _ in range(5):
        r = seq_filter(log_likes)
        r.marginal_log_lik.block_until_ready()
    t_seq = (time.perf_counter() - t0) / 5.0 * 1000.0
    print(f"Sequential HMM filter (GPU): {t_seq:7.2f} ms")

    t0 = time.perf_counter()
    for _ in range(5):
        r = par_filter(log_likes)
        r.marginal_log_lik.block_until_ready()
    t_par = (time.perf_counter() - t0) / 5.0 * 1000.0
    print(f"Parallel Associative HMM filter (GPU): {t_par:7.2f} ms ({t_seq/t_par:5.2f}x speedup)")

if __name__ == "__main__":
    test_parallel_hmm()
