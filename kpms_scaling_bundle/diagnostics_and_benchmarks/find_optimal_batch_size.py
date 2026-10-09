import os
import sys
import time

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax
import jax.numpy as jnp
import jax.random as jr

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")

def probe_batch_size(bs, N=1004, T=10030, k=17, d=2, latent_dim=10, nlags=3, num_states=100):
    """
    Tests whether a given batch size `bs` runs without OOM on simulated full-dataset dimensions.
    Returns (success: bool, elapsed_seconds: float)
    """
    print(f"\n[Probe] Testing batch_size = {bs} on {N} videos (bs slice = {min(bs, N)})...")
    seed = jr.PRNGKey(42)

    slice_n = min(bs, N)
    try:
        from utils.resample_batching import _jit_resample_discrete_batch, _jit_resample_x_batch

        # Allocate one batch test on GPU
        Y = jr.normal(jr.fold_in(seed, 1), (slice_n, T, k, d))
        mask = jnp.ones((slice_n, T), dtype=bool)
        v = jr.normal(jr.fold_in(seed, 2), (slice_n, T, d))
        h = jr.uniform(jr.fold_in(seed, 3), (slice_n, T), minval=-np.pi, maxval=np.pi)
        s = jnp.ones((slice_n, T, k), dtype=jnp.float64)
        z = jr.randint(jr.fold_in(seed, 4), (slice_n, T - nlags), 0, num_states)
        Cd = jr.normal(jr.fold_in(seed, 5), ((k - 1) * d, latent_dim + 1))
        sigmasq = jnp.ones(k, dtype=jnp.float64) * 0.1
        Ab = jr.normal(jr.fold_in(seed, 6), (num_states, latent_dim, nlags * latent_dim + 1)) * 0.01
        Q = jnp.tile(jnp.eye(latent_dim, dtype=jnp.float64)[None, ...], (num_states, 1, 1))
        pi = jnp.ones((num_states, num_states), dtype=jnp.float64) / num_states
        x = jr.normal(jr.fold_in(seed, 7), (slice_n, T, latent_dim))

        # Warmup and timed run
        t0 = time.perf_counter()
        _ = _jit_resample_discrete_batch(seed, x, mask, Ab, Q, pi).block_until_ready()
        _ = _jit_resample_x_batch(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q).block_until_ready()
        t_tot = time.perf_counter() - t0

        # Clean up
        del Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, pi, x
        jax.clear_caches()
        print(f"[Probe] SUCCESS: bs = {bs} completed in {t_tot:.2f}s (Warmup+Run)")
        return True, t_tot
    except Exception as e:
        err_msg = str(e)
        print(f"[Probe] FAILED / OOM for bs = {bs}: {err_msg[:120]}...")
        jax.clear_caches()
        return False, None

def binary_search_optimal_bs(low=1, high=1004):
    """
    Binary searches between lower bound (bs=1) and upper bound (bs=1004)
    to find the maximum feasible batch size that fits in 24GB VRAM.
    """
    print("=" * 70)
    print("      BINARY SEARCH FOR OPTIMAL GPU BATCH SIZE ON RTX 4090 (24GB)")
    print(f"      Lower Bound: {low} | Upper Bound: {high}")
    print("=" * 70)

    best_bs = low
    # First check upper bound directly
    ok, _ = probe_batch_size(high)
    if ok:
        print(f"\n[Result] Full upper bound bs = {high} fits directly without batching!")
        return high

    # Binary search
    curr_low = low
    curr_high = high - 1

    while curr_low <= curr_high:
        mid = (curr_low + curr_high) // 2
        ok, _ = probe_batch_size(mid)
        if ok:
            best_bs = mid
            print(f"--> bs = {mid} PASSED. Trying higher...")
            curr_low = mid + 1
        else:
            print(f"--> bs = {mid} FAILED (OOM). Trying lower...")
            curr_high = mid - 1

    print("\n" + "=" * 70)
    print(f" OPTIMAL MAX BATCH SIZE FOUND: bs = {best_bs}")
    print("=" * 70)
    return best_bs

if __name__ == "__main__":
    best_bs = binary_search_optimal_bs(low=1, high=1004)
