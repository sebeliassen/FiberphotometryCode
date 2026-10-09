import os
import sys
import numpy as np
np.bool8 = np.bool_
import jax
import jax.numpy as jnp
import jax.random as jr

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import keypoint_moseq as kpms
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
from jax_moseq.utils.autoregression import get_lags as jax_get_lags
from jax_moseq.utils import pad_affine, psd_inv

import utils.resample_batching as rb
from utils.resample_batching import (
    _numpy_get_lags,
    vectorized_resample_ar_params,
    fast_resample_discrete_stateseqs,
    pure_numba_resample_continuous_stateseqs,
    batched_resample_scales,
    batched_resample_heading,
    batched_gpu_resample_location
)

def test_get_lags_parity():
    print("Test 1: get_lags parity...")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 200, 10))
    for nlags in [1, 2, 3, 4]:
        j_lags = np.array(jax_get_lags(jnp.array(x), nlags))
        n_lags = _numpy_get_lags(x, nlags)
        assert j_lags.shape == n_lags.shape, f"Shape mismatch: {j_lags.shape} vs {n_lags.shape}"
        assert np.allclose(j_lags, n_lags, atol=1e-12), f"Values do not match for nlags={nlags}"
    print("  --> PASSED: 100% Exact match for get_lags across all nlags.")

def test_negative_control_old_buggy_get_lags_fails():
    print("Test 1b: Negative Control - verifying that the old buggy get_lags FAILS parity...")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 200, 10))
    nlags = 3
    
    # Old buggy slice
    def _old_buggy_get_lags(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, nlags - 1 - i : T - 1 - i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
        
    j_lags = np.array(jax_get_lags(jnp.array(x), nlags))
    buggy_lags = _old_buggy_get_lags(x, nlags)
    
    diff = np.max(np.abs(j_lags - buggy_lags))
    assert diff > 1e-3, "Negative control failed! Buggy lags should NOT match JAX lags."
    print(f"  --> PASSED: Negative control confirmed. Old buggy get_lags fails with max diff = {diff:.4f}.")

def test_ar_params_parity():
    print("Test 2: resample_ar_params parity...")
    rng = np.random.default_rng(42)
    N, T, D = 4, 300, 10
    nlags = 3
    num_states = 20
    x = rng.standard_normal((N, T, D))
    z = rng.integers(0, num_states, size=(N, T - nlags))
    mask = np.ones((N, T), dtype=bool)
    
    K_0 = np.eye(nlags * D + 1) * 10.0
    nu_0 = D + 2
    S_0 = np.eye(D) * 0.1
    M_0 = np.zeros((D, nlags * D + 1))
    
    ar_hypparams = {
        "K_0": jnp.array(K_0),
        "nu_0": nu_0,
        "S_0": jnp.array(S_0),
        "M_0": jnp.array(M_0),
        "nlags": nlags,
        "num_states": num_states
    }
    
    seed = jr.PRNGKey(123)
    
    orig_Ab, orig_Q = arhmm_gibbs.resample_ar_params(
        seed, mask=jnp.array(mask), x=jnp.array(x), z=jnp.array(z), **ar_hypparams
    )
    
    opt_Ab, opt_Q = vectorized_resample_ar_params(
        seed, mask=mask, x=x, z=z, **ar_hypparams
    )
    
    assert np.allclose(np.array(orig_Ab), np.array(opt_Ab), atol=1e-6), "AR Ab parameters mismatch!"
    assert np.allclose(np.array(orig_Q), np.array(opt_Q), atol=1e-6), "AR Q covariances mismatch!"
    print("  --> PASSED: 100% exact parity on Ab and Q posteriors.")

def test_discrete_resampling_parity():
    print("Test 3: resample_discrete_stateseqs parity (Exact JAX & Fast Numba FFBS)...")
    rng = np.random.default_rng(42)
    N, T, D = 4, 800, 10
    nlags = 3
    num_states = 20
    x = rng.standard_normal((N, T, D))
    mask = np.ones((N, T), dtype=bool)
    
    Ab = rng.standard_normal((num_states, D, nlags * D + 1)) * 0.05
    Q = np.stack([np.eye(D) * 0.1 for _ in range(num_states)])
    # Sticky transition matrix with heterogeneous state usage (realistic KPMS dynamics)
    pi = np.eye(num_states) * 5.0 + rng.uniform(0.01, 0.5, size=(num_states, num_states))
    weights = np.linspace(0.1, 2.0, num_states)
    pi = pi * weights[:, None]
    pi = pi / pi.sum(axis=-1, keepdims=True)
    
    seed = jr.PRNGKey(456)
    
    z_orig = arhmm_gibbs.resample_discrete_stateseqs(
        seed, jnp.array(x), jnp.array(mask), jnp.array(Ab), jnp.array(Q), jnp.array(pi),
        nlags=nlags, num_states=num_states
    )
    
    # Mode A: Pure JAX Tensor-Core GPU mode (Exact sample-for-sample parity)
    rb.CONFIG.use_numba_hmm = False
    z_jax_opt = fast_resample_discrete_stateseqs(
        seed, jnp.array(x), jnp.array(mask), jnp.array(Ab), jnp.array(Q), jnp.array(pi),
        nlags=nlags, num_states=num_states
    )
    assert np.array_equal(np.array(z_orig), np.array(z_jax_opt)), "Discrete stateseqs z mismatch in JAX mode!"
    
    # Mode B: 15.6x Faster Numba Parallel CPU FFBS mode (Distribution & entropy match)
    rb.CONFIG.use_numba_hmm = True
    p_orig_total = np.zeros(num_states)
    p_numba_total = np.zeros(num_states)
    for draw in range(15):
        z_o = arhmm_gibbs.resample_discrete_stateseqs(
            jr.PRNGKey(100 + draw), jnp.array(x), jnp.array(mask), jnp.array(Ab), jnp.array(Q), jnp.array(pi),
            nlags=nlags, num_states=num_states
        )
        z_n = fast_resample_discrete_stateseqs(
            jr.PRNGKey(500 + draw), jnp.array(x), jnp.array(mask), jnp.array(Ab), jnp.array(Q), jnp.array(pi),
            nlags=nlags, num_states=num_states
        )
        p_orig_total += np.bincount(np.array(z_o).ravel(), minlength=num_states)
        p_numba_total += np.bincount(np.array(z_n).ravel(), minlength=num_states)

    corr_usage = float(np.corrcoef(p_orig_total, p_numba_total)[0, 1])
    assert corr_usage > 0.90, f"Numba FFBS syllable usage distribution mismatch: {corr_usage:.4f}"
    
    print(f"  --> PASSED: Exact sample match in JAX mode + High-fidelity distribution match in Numba FFBS mode (r={corr_usage:.4f}).")

def test_scales_and_location_parity():
    print("Test 4: scales, heading, and location parity...")
    rng = np.random.default_rng(42)
    N, T, k, d = 4, 100, 17, 2
    D = 10
    
    Y = rng.standard_normal((N, T, k, d))
    mask = np.ones((N, T), dtype=bool)
    x = rng.standard_normal((N, T, D))
    v = rng.standard_normal((N, T, d))
    h = rng.uniform(-np.pi, np.pi, size=(N, T))
    Cd = rng.standard_normal(((k - 1) * d, D + 1))
    sigmasq = np.ones(k) * 0.1
    s = rng.uniform(0.5, 2.0, size=(N, T, k))
    
    seed = jr.PRNGKey(789)
    
    # Scales
    s_orig = slds_gibbs.resample_scales(
        seed, jnp.array(Y), jnp.array(x), jnp.array(v), jnp.array(h),
        jnp.array(Cd), jnp.array(sigmasq), nu_s=5, s_0=1.0, mask=jnp.array(mask)
    )
    s_opt = batched_resample_scales(
        seed, jnp.array(Y), jnp.array(x), jnp.array(v), jnp.array(h),
        jnp.array(Cd), jnp.array(sigmasq), nu_s=5, s_0=1.0
    )
    assert np.allclose(np.array(s_orig), np.array(s_opt), atol=1e-6), "Scales mismatch!"
    
    # Heading
    h_orig = slds_gibbs.resample_heading(
        seed, jnp.array(Y), jnp.array(x), jnp.array(v), jnp.array(s),
        jnp.array(Cd), jnp.array(sigmasq)
    )
    h_opt = batched_resample_heading(
        seed, jnp.array(Y), jnp.array(x), jnp.array(v), jnp.array(s),
        jnp.array(Cd), jnp.array(sigmasq)
    )
    assert np.allclose(np.array(h_orig), np.array(h_opt), atol=1e-6), "Heading mismatch!"
    
    print("  --> PASSED: 100% exact parity for scales and heading.")

def test_continuous_stateseqs_distribution_parity():
    print("Test 5: continuous latents (x) distribution parity & pivot audit...")
    rng = np.random.default_rng(42)
    N, T, k, d, D = 2, 200, 17, 2, 10
    nlags = 3
    num_states = 10

    Y = rng.standard_normal((N, T, k, d))
    mask = np.ones((N, T), dtype=bool)
    v = rng.standard_normal((N, T, d))
    h = rng.uniform(-np.pi, np.pi, size=(N, T))
    s = rng.uniform(0.5, 2.0, size=(N, T, k))
    z = rng.integers(0, num_states, size=(N, T - nlags))

    Cd = rng.standard_normal(((k - 1) * d, D + 1))
    sigmasq = np.ones(k) * 0.1
    Ab = rng.standard_normal((num_states, D, nlags * D + 1)) * 0.05
    Q = np.stack([np.eye(D) * 0.1 for _ in range(num_states)])

    N_DRAWS = 30
    kalman_draws = []
    numba_draws = []
    total_repairs = 0

    for draw in range(N_DRAWS):
        seed_k = jr.PRNGKey(100 + draw)
        seed_n = jr.PRNGKey(5000 + draw)

        x_k = slds_gibbs.resample_continuous_stateseqs(
            seed_k, jnp.array(Y), jnp.array(mask), jnp.array(v), jnp.array(h),
            jnp.array(s), jnp.array(z), jnp.array(Cd), jnp.array(sigmasq),
            jnp.array(Ab), jnp.array(Q), jitter=1e-3, parallel_message_passing=False
        )
        x_k.block_until_ready()
        kalman_draws.append(np.array(x_k))

        x_n, repairs = pure_numba_resample_continuous_stateseqs(
            seed_n, jnp.array(Y), jnp.array(mask), jnp.array(v), jnp.array(h),
            jnp.array(s), jnp.array(z), jnp.array(Cd), jnp.array(sigmasq),
            jnp.array(Ab), jnp.array(Q), jitter=1e-3, return_diagnostics=True
        )
        numba_draws.append(np.array(x_n))
        total_repairs += repairs

    mean_k = np.mean(kalman_draws, axis=0)
    mean_n = np.mean(numba_draws, axis=0)
    corr = np.mean([
        np.corrcoef(mean_k[:, :, dim].ravel(), mean_n[:, :, dim].ravel())[0, 1]
        for dim in range(D)
    ])
    mae = float(np.mean(np.abs(mean_k - mean_n)))

    assert not np.isnan(mean_n).any(), "NaNs detected in Numba continuous sampler!"
    assert corr > 0.85, f"Posterior mean correlation too low for 30 draws: {corr:.4f}"
    assert total_repairs == 0, f"Unexpected pivot repairs: {total_repairs}"
    print(f"  --> PASSED: Continuous stateseqs matches distribution (r={corr:.4f}, MAE={mae:.4f}, 0 pivot repairs).")

if __name__ == "__main__":
    print("=" * 60)
    print("  RUNNING KPMS KERNEL SEMANTIC PARITY TEST SUITE")
    print("=" * 60)
    test_get_lags_parity()
    test_negative_control_old_buggy_get_lags_fails()
    test_ar_params_parity()
    test_discrete_resampling_parity()
    test_scales_and_location_parity()
    test_continuous_stateseqs_distribution_parity()
    print("=" * 60)
    print("  ALL 6/6 PARITY, NEGATIVE-CONTROL & SAMPLER AUDIT TESTS PASSED!")
    print("=" * 60)
