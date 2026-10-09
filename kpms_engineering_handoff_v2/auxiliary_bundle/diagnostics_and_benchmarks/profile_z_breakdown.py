import os
import sys
import time

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers/keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax
import jax.numpy as jnp
import jax.random as jr
import keypoint_moseq as kpms
import utils.resample_batching as rb
from dynamax.hidden_markov_model.inference import hmm_filter, hmm_posterior_sample
from jax_moseq.utils import pad_affine, psd_inv
from jax_moseq.utils.autoregression import get_lags

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
TEST_MODEL_NAME = "test_2_sessions_model"
LIVE_LOG_PATH = "/tmp/kpms_live.log"

def log(msg):
    print(msg, flush=True)
    try:
        with open(LIVE_LOG_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def sync(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for item in x:
            sync(item)
    return x

def main():
    log("=" * 80)
    log("     DISCRETE STATES (z) RESAMPLING: SUB-COMPONENT TIMING PROFILER")
    log("=" * 80)

    # 1. Load real data from checkpoint
    log(f"[1/4] Loading parameters and dataset from {TEST_MODEL_NAME}...")
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, TEST_MODEL_NAME)

    N_SEQS = 4
    x = jnp.array(model["states"]["x"][:N_SEQS])
    mask = jnp.array(data["mask"][:N_SEQS])
    Ab = jnp.array(model["params"]["Ab"])
    Q = jnp.array(model["params"]["Q"])
    pi = jnp.array(model["params"]["pi"])

    N, T, D = x.shape
    num_states = Ab.shape[0]
    nlags = (Ab.shape[-1] - 1) // D

    log(f"      Dataset: N={N} sequences, T={T:,} frames, K={num_states} states, D={D} latents, lags={nlags}")

    # =========================================================================
    # Step 1: Sub-Component Breakdown of z Resampling
    # =========================================================================
    log("\n[2/4] Profiling sub-components: Emissions vs Forward Filtering vs Backward Sampling...")

    # A. Covariance factors
    @jax.jit
    def precompute_q_factors(Q):
        L_Q = jnp.linalg.cholesky(Q)
        half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
        const = (0.5 * D * np.log(2.0 * np.pi)) + half_log_det_Q
        Q_inv = jax.vmap(psd_inv)(Q)
        return L_Q, const, Q_inv

    # B. Emission computation
    @jax.jit
    def compute_emissions(x, Ab, Q_inv, const):
        lags = get_lags(x, nlags)
        x_lags = pad_affine(lags)
        x_target = x[:, nlags:, :]
        mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
        diff = x_target[:, :, None, :] - mu_all
        quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
        log_likelihoods = -0.5 * quad - const[None, None, :]
        return log_likelihoods

    # C. Forward filter only
    initial_dist = jnp.ones(num_states, dtype=jnp.float64) / num_states

    @jax.jit
    def forward_filter_all(log_liks, mask):
        def filter_one(ll_i, m_i):
            masked_ll = ll_i * m_i[nlags:, None]
            post = hmm_filter(initial_dist, pi, masked_ll)
            return post.filtered_probs
        return jax.vmap(filter_one)(log_liks, mask)

    # D. Backward sampling only
    @jax.jit
    def backward_sample_all(seed, filtered_probs):
        def sample_one(seed_i, probs_i):
            T_eff = probs_i.shape[0]
            keys = jr.split(seed_i, T_eff)
            last_state = jr.choice(keys[-1], a=num_states, p=probs_i[-1])
            def _step(carry, args):
                next_s = carry
                t, subkey, f_probs = args
                smoothed = f_probs * pi[:, next_s]
                smoothed = smoothed / smoothed.sum()
                s = jr.choice(subkey, a=num_states, p=smoothed)
                return s, s
            _, states = jax.lax.scan(
                _step, last_state,
                (jnp.arange(1, T_eff), keys[:-1], probs_i[:-1]),
                reverse=True
            )
            return jnp.concatenate([states, jnp.array([last_state])])
        return jax.vmap(sample_one)(jr.split(seed, N), filtered_probs)

    # Warm-up compiles
    L_Q, const, Q_inv = precompute_q_factors(Q)
    sync((L_Q, const, Q_inv))
    ll = compute_emissions(x, Ab, Q_inv, const)
    sync(ll)
    f_probs = forward_filter_all(ll, mask)
    sync(f_probs)
    dummy_seed = jr.PRNGKey(0)
    z_back = backward_sample_all(dummy_seed, f_probs)
    sync(z_back)

    # Benchmark sub-steps across 10 iterations
    N_BENCH = 10

    # 1. Covariance factors
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        out = precompute_q_factors(Q)
        sync(out)
    t_cov = (time.perf_counter() - t0) / N_BENCH

    # 2. Emissions (likelihoods)
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        ll = compute_emissions(x, Ab, Q_inv, const)
        sync(ll)
    t_emiss = (time.perf_counter() - t0) / N_BENCH

    # 3. Forward filter
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        f_probs = forward_filter_all(ll, mask)
        sync(f_probs)
    t_filt = (time.perf_counter() - t0) / N_BENCH

    # 4. Backward sample
    t0 = time.perf_counter()
    for i in range(N_BENCH):
        seed_i = jr.fold_in(dummy_seed, i)
        z_back = backward_sample_all(seed_i, f_probs)
        sync(z_back)
    t_back = (time.perf_counter() - t0) / N_BENCH

    # 5. Full end-to-end current kernel
    t0 = time.perf_counter()
    for i in range(N_BENCH):
        seed_i = jr.fold_in(dummy_seed, i)
        z_e2e = rb.fast_resample_discrete_stateseqs(seed_i, x, mask, Ab, Q, pi)
        sync(z_e2e)
    t_full = (time.perf_counter() - t0) / N_BENCH

    # Intermediate memory estimation
    T_eff = T - nlags
    intermediate_mu_bytes = N * T_eff * num_states * D * 8
    intermediate_diff_bytes = N * T_eff * num_states * D * 8
    intermediate_ll_bytes = N * T_eff * num_states * 8
    f_probs_bytes = N * T_eff * num_states * 8

    log("\n" + "=" * 80)
    log(f"{'DISCRETE STATE (z) SUB-COMPONENT PROFILING RESULTS':^80}")
    log("=" * 80)
    log(f"{'Component':<35} | {'Time (s)':<12} | {'% of z Time':<12} | {'Memory (MB)':<12}")
    log("-" * 80)
    log(f"{'1. Covariance Factorization (Q_inv)':<35} | {t_cov:10.4f} s | {100 * t_cov / t_full:10.1f}% | {'< 1 MB':<12}")
    log(f"{'2. Emission Evaluation (mu, diff, quad)':<35} | {t_emiss:10.4f} s | {100 * t_emiss / t_full:10.1f}% | {(intermediate_mu_bytes + intermediate_diff_bytes + intermediate_ll_bytes)/1e6:10.1f} MB")
    log(f"{'3. Forward Filtering (hmm_filter)':<35} | {t_filt:10.4f} s | {100 * t_filt / t_full:10.1f}% | {f_probs_bytes/1e6:10.1f} MB")
    log(f"{'4. Backward Sampling (lax.scan)':<35} | {t_back:10.4f} s | {100 * t_back / t_full:10.1f}% | {'< 1 MB':<12}")
    log("-" * 80)
    log(f"{'Sum of Components':<35} | {t_cov + t_emiss + t_filt + t_back:10.4f} s | {100 * (t_cov + t_emiss + t_filt + t_back) / t_full:10.1f}% | -")
    log(f"{'Full JAX Pipeline (fast_resample)':<35} | {t_full:10.4f} s | {'100.0%':>10} | -")
    log("=" * 80)

    # =========================================================================
    # Step 2: Sequence Batch Sweep & Covariance Factor Reuse
    # =========================================================================
    log("\n[3/4] Benchmarking Sequence Batch Sizes (Reusing Covariance Factors)...")
    log(f"{'Batch Size (BS)':<20} | {'Time per Batch (s)':<20} | {'Effective Time for N=4 (s)':<25}")
    log("-" * 80)

    for bs in [1, 2, 4]:
        @jax.jit
        def run_batch_with_reuse(seed_b, x_b, mask_b):
            ll_b = compute_emissions(x_b, Ab, Q_inv, const)
            def sample_one(seed_i, ll_i, mask_i):
                masked_ll = ll_i * mask_i[nlags:, None]
                _, z = hmm_posterior_sample(seed_i, initial_dist, pi, masked_ll)
                return z
            return jax.vmap(sample_one)(jr.split(seed_b, x_b.shape[0]), ll_b, mask_b)

        # Warm-up
        x_sub = x[:bs]
        mask_sub = mask[:bs]
        sync(run_batch_with_reuse(dummy_seed, x_sub, mask_sub))

        t0 = time.perf_counter()
        for i in range(N_BENCH):
            s_i = jr.fold_in(dummy_seed, i)
            out = run_batch_with_reuse(s_i, x_sub, mask_sub)
            sync(out)
        t_batch = (time.perf_counter() - t0) / N_BENCH
        eff_time_n4 = t_batch * (N / bs)
        log(f"{bs:<20} | {t_batch:18.4f} s | {eff_time_n4:23.4f} s")

    log("=" * 80)
    log("[4/4] Sub-component profiling completed successfully!\n")

if __name__ == "__main__":
    main()
