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
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jax_moseq.utils import pad_affine, psd_inv
from jax_moseq.utils.autoregression import get_lags
from numba import njit, prange

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

# =============================================================================
# High-Performance Numba Parallel Forward-Filtering Backward-Sampling (FFBS)
# =============================================================================

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
def _numba_ffbs_single_sequence(log_likes, pi, mask, random_uniforms, z_out, P_workspace):
    """
    Run forward-filtering and backward-sampling for one sequence.
    T: time steps, K: number of states.
    log_likes: (T, K)
    pi: (K, K) where pi[j, k] = P(z_t = k | z_{t-1} = j)
    mask: (T,)
    random_uniforms: (T,)
    z_out: (T,) int32
    P_workspace: (T, K) float64
    """
    T, K = log_likes.shape

    # 1. Forward pass t = 0
    max_ll = -1e300
    for k in range(K):
        if log_likes[0, k] > max_ll:
            max_ll = log_likes[0, k]

    sum_p = 0.0
    for k in range(K):
        if mask[0]:
            val = np.exp(log_likes[0, k] - max_ll)
        else:
            val = 1.0
        P_workspace[0, k] = val
        sum_p += val

    inv_sum = 1.0 / sum_p
    for k in range(K):
        P_workspace[0, k] *= inv_sum

    # Forward pass t = 1 ... T - 1
    pred = np.empty(K, dtype=np.float64)
    for t in range(1, T):
        # Matrix-vector multiplication: pred_k = sum_j P[t-1, j] * pi[j, k]
        for k in range(K):
            s = 0.0
            for j in range(K):
                s += P_workspace[t - 1, j] * pi[j, k]
            pred[k] = s

        if mask[t]:
            # Scale-stabilized emission multiplication
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
            # Masked frame: pure prediction
            sum_p = 0.0
            for k in range(K):
                val = pred[k]
                P_workspace[t, k] = val
                sum_p += val

        inv_sum = 1.0 / (sum_p + 1e-300)
        for k in range(K):
            P_workspace[t, k] *= inv_sum

    # 2. Backward sampling
    # Sample last state at T - 1
    last_state = _numba_sample_categorical(P_workspace[T - 1], random_uniforms[T - 1])
    z_out[T - 1] = last_state

    # Sample t = T - 2 down to 0
    q = np.empty(K, dtype=np.float64)
    for t in range(T - 2, -1, -1):
        next_s = z_out[t + 1]
        sum_q = 0.0
        for j in range(K):
            val = P_workspace[t, j] * pi[j, next_s]
            q[j] = val
            sum_q += val

        inv_sum_q = 1.0 / (sum_q + 1e-300)
        for j in range(K):
            q[j] *= inv_sum_q

        z_out[t] = _numba_sample_categorical(q, random_uniforms[t])

@njit(fastmath=True, parallel=True)
def _numba_ffbs_all_parallel(log_likes_all, pi, mask_all, uniforms_all, z_out_all, P_all):
    N = log_likes_all.shape[0]
    for i in prange(N):
        _numba_ffbs_single_sequence(
            log_likes_all[i], pi, mask_all[i],
            uniforms_all[i], z_out_all[i], P_all[i]
        )

def main():
    log("=" * 80)
    log("   DISCRETE STATE (z) ACCELERATION: NUMBA CPU FFBS vs JAX GPU BENCHMARK")
    log("=" * 80)

    # 1. Load real data
    log(f"[1/4] Loading parameters and dataset from {TEST_MODEL_NAME}...")
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, TEST_MODEL_NAME)

    N_SEQS = 4
    x_jnp = jnp.array(model["states"]["x"][:N_SEQS])
    mask_jnp = jnp.array(data["mask"][:N_SEQS])
    Ab_jnp = jnp.array(model["params"]["Ab"])
    Q_jnp = jnp.array(model["params"]["Q"])
    pi_jnp = jnp.array(model["params"]["pi"])

    N, T, D = x_jnp.shape
    num_states = Ab_jnp.shape[0]
    nlags = (Ab_jnp.shape[-1] - 1) // D
    T_eff = T - nlags

    log(f"      Dataset: N={N} seqs, T={T:,} frames, K={num_states} states, D={D} latents")

    # 2. Compute emissions on GPU (fast 7.7 ms kernel)
    log("\n[2/4] Computing GPU emissions & precomputing covariance factors...")
    L_Q = jnp.linalg.cholesky(Q_jnp)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = (0.5 * D * np.log(2.0 * np.pi)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(Q_jnp)

    @jax.jit
    def compute_emissions(x, Ab, Q_inv, const):
        lags = get_lags(x, nlags)
        x_lags = pad_affine(lags)
        x_target = x[:, nlags:, :]
        mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
        diff = x_target[:, :, None, :] - mu_all
        quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
        return -0.5 * quad - const[None, None, :]

    log_likes_gpu = compute_emissions(x_jnp, Ab_jnp, Q_inv, const)
    log_likes_gpu.block_until_ready()

    # 3. Benchmark Upstream JAX GPU Sampler
    log("\n[3/4] Benchmarking JAX GPU HMM Sampler (Forward Filtering + Backward Sampling)...")
    initial_dist = jnp.ones(num_states, dtype=jnp.float64) / num_states

    @jax.jit
    def jax_gpu_sample_batch(seed, ll, mask):
        def sample_one(seed_i, ll_i, mask_i):
            masked_ll = ll_i * mask_i[nlags:, None]
            _, z = hmm_posterior_sample(seed_i, initial_dist, pi_jnp, masked_ll)
            return z
        return jax.vmap(sample_one)(jr.split(seed, N), ll, mask)

    # Warmup
    dummy_seed = jr.PRNGKey(42)
    z_jax = jax_gpu_sample_batch(dummy_seed, log_likes_gpu, mask_jnp)
    z_jax.block_until_ready()

    N_BENCH = 10
    t0 = time.perf_counter()
    for i in range(N_BENCH):
        s_i = jr.fold_in(dummy_seed, i)
        out = jax_gpu_sample_batch(s_i, log_likes_gpu, mask_jnp)
        out.block_until_ready()
    t_jax_gpu = (time.perf_counter() - t0) / N_BENCH
    log(f"      JAX GPU HMM Sampler Time: {t_jax_gpu * 1000:8.2f} ms ({t_jax_gpu:.4f} s)")

    # 4. Benchmark Numba Parallel CPU FFBS Sampler (including Host/Device Transfers)
    log("\n[4/4] Benchmarking Numba Parallel CPU FFBS Sampler (with Transfer Overhead)...")
    pi_np = np.asarray(pi_jnp, dtype=np.float64)
    mask_np = np.asarray(mask_jnp[:, nlags:], dtype=bool)

    # Pre-allocate workspaces
    P_workspace_all = np.empty((N, T_eff, num_states), dtype=np.float64)
    z_out_all = np.empty((N, T_eff), dtype=np.int32)
    uniforms_all = np.empty((N, T_eff), dtype=np.float64)

    # Warmup compile
    dummy_uniforms = np.random.uniform(0.0, 1.0, size=(N, T_eff))
    ll_np = np.asarray(log_likes_gpu)
    _numba_ffbs_all_parallel(ll_np, pi_np, mask_np, dummy_uniforms, z_out_all, P_workspace_all)

    # Benchmark full pipeline with transfer: Device -> Host -> Numba FFBS -> Host -> Device
    rng = np.random.default_rng(123)
    t_transfer_d2h_list = []
    t_numba_core_list = []
    t_transfer_h2d_list = []

    t0 = time.perf_counter()
    for i in range(N_BENCH):
        # 1. Device to Host transfer
        t_d2h_start = time.perf_counter()
        ll_cpu = np.asarray(log_likes_gpu)
        t_transfer_d2h_list.append(time.perf_counter() - t_d2h_start)

        # 2. Random generation + Numba Parallel FFBS
        t_core_start = time.perf_counter()
        uniforms_all[:] = rng.uniform(0.0, 1.0, size=(N, T_eff))
        _numba_ffbs_all_parallel(ll_cpu, pi_np, mask_np, uniforms_all, z_out_all, P_workspace_all)
        t_numba_core_list.append(time.perf_counter() - t_core_start)

        # 3. Host to Device transfer
        t_h2d_start = time.perf_counter()
        z_gpu = jnp.asarray(z_out_all)
        z_gpu.block_until_ready()
        t_transfer_h2d_list.append(time.perf_counter() - t_h2d_start)

    t_numba_total = (time.perf_counter() - t0) / N_BENCH
    t_d2h = np.mean(t_transfer_d2h_list)
    t_core = np.mean(t_numba_core_list)
    t_h2d = np.mean(t_transfer_h2d_list)

    log(f"      D2H Transfer (Log-likes GPU->CPU): {t_d2h * 1000:8.2f} ms")
    log(f"      Numba Parallel FFBS (CPU Compute): {t_core * 1000:8.2f} ms")
    log(f"      H2D Transfer (z CPU->GPU):         {t_h2d * 1000:8.2f} ms")
    log(f"      Total Numba CPU Pipeline Time:     {t_numba_total * 1000:8.2f} ms ({t_numba_total:.4f} s)")

    speedup_core = t_jax_gpu / t_core
    speedup_e2e = t_jax_gpu / t_numba_total

    # Statistical distribution check: compare syllable duration distributions
    z_jax_np = np.array(z_jax)
    z_numba_np = np.array(z_out_all)

    # Compute usage entropy and median durations
    from scipy.stats import entropy
    counts_jax = np.bincount(z_jax_np.ravel(), minlength=num_states)
    p_jax = counts_jax / counts_jax.sum()
    ent_jax = float(entropy(p_jax[p_jax > 0]))

    counts_numba = np.bincount(z_numba_np.ravel(), minlength=num_states)
    p_numba = counts_numba / counts_numba.sum()
    ent_numba = float(entropy(p_numba[p_numba > 0]))

    # Syllable distribution correlation
    corr_usage = float(np.corrcoef(p_jax, p_numba)[0, 1])

    log("\n" + "=" * 80)
    log(f"{'HMM SAMPLER ACCELERATION RESULTS & COMPARISON':^80}")
    log("=" * 80)
    log(f"{'Metric / Property':<40} | {'JAX GPU':<16} | {'Numba CPU':<16}")
    log("-" * 80)
    log(f"{'Sampler Compute Time':<40} | {t_jax_gpu * 1000:13.2f} ms | {t_core * 1000:13.2f} ms ({speedup_core:5.1f}x)")
    log(f"{'End-to-End Time (with PCIe Transfers)':<40} | {t_jax_gpu * 1000:13.2f} ms | {t_numba_total * 1000:13.2f} ms ({speedup_e2e:5.1f}x)")
    log(f"{'Syllable Usage Correlation (r)':<40} | {'1.0000 (Ref)':<16} | {corr_usage:13.4f}")
    log(f"{'State Entropy (bits)':<40} | {ent_jax:13.4f}    | {ent_numba:13.4f}")
    log("=" * 80)

    if speedup_e2e > 1.0:
        log(f"\nSUCCESS: Numba CPU FFBS delivers a {speedup_e2e:.2f}x end-to-end speedup over JAX GPU!\n")
    else:
        log("\nRESULT: JAX GPU remains faster than CPU + PCIe transfer.\n")

if __name__ == "__main__":
    main()
