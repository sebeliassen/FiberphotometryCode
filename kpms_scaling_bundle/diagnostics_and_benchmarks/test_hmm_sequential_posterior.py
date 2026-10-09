#!/usr/bin/env python3
"""
diagnostics/test_hmm_sequential_posterior.py

Rigorous statistical verification of the Numba parallel FFBS discrete state sampler:
1. Tests temporal sequence distribution P(z_t = k | y_1:T), not just marginal syllable counts.
2. Tests 2-step transition distribution P(z_t = j | z_{t-1} = i, y_1:T).
3. Tests masked/missing frame behavior (pure transition propagation).
4. Compares empirical Monte Carlo distributions against exact analytic forward-backward posteriors.
"""

import os
import sys

import numpy as np

# Add parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.resample_batching import _numba_ffbs_single_sequence


def compute_exact_forward_backward(log_likes, pi, mask):
    """
    Compute exact analytic posterior marginals P(z_t = k | y) and 
    pairwise transition posteriors P(z_{t-1}=i, z_t=j | y) via exact forward-backward.
    """
    T, K = log_likes.shape
    alpha = np.zeros((T, K), dtype=np.float64)
    beta = np.zeros((T, K), dtype=np.float64)
    scale = np.zeros(T, dtype=np.float64)

    # Initial state distribution (uniform prior if not stationary)
    p0 = np.ones(K) / K

    # Forward pass
    ll0 = log_likes[0] if mask[0] else np.zeros(K)
    c0 = np.exp(ll0 - np.max(ll0)) * p0
    scale[0] = np.sum(c0)
    alpha[0] = c0 / scale[0]

    for t in range(1, T):
        pred = alpha[t - 1] @ pi
        ll_t = log_likes[t] if mask[t] else np.zeros(K)
        c_t = np.exp(ll_t - np.max(ll_t)) * pred
        scale[t] = np.sum(c_t)
        alpha[t] = c_t / scale[t]

    # Backward pass
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        ll_next = log_likes[t + 1] if mask[t + 1] else np.zeros(K)
        b_next = np.exp(ll_next - np.max(ll_next)) * beta[t + 1]
        b_t = pi @ b_next
        beta[t] = b_t / scale[t + 1]

    # Posterior marginals P(z_t = k | y)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=-1, keepdims=True)

    # Pairwise posteriors xi[t, i, j] = P(z_{t-1}=i, z_t=j | y)
    xi = np.zeros((T - 1, K, K), dtype=np.float64)
    for t in range(T - 1):
        ll_next = log_likes[t + 1] if mask[t + 1] else np.zeros(K)
        b_next = np.exp(ll_next - np.max(ll_next)) * beta[t + 1]
        pair = (alpha[t, :, None] * pi) * b_next[None, :]
        xi[t] = pair / np.sum(pair)

    return gamma, xi


def run_temporal_posterior_test():
    print("=" * 65)
    print("  RUNNING HMM TEMPORAL POSTERIOR & SEQUENCE DISTRIBUTION TEST")
    print("=" * 65)

    np.random.seed(42)
    T = 60
    K = 4

    # 1. Construct a structured transition matrix with distinct temporal dynamics
    # State 0 -> 1 -> 2 -> 3 -> 0 (a strong directional cyclic structure)
    pi = np.array([
        [0.10, 0.70, 0.10, 0.10],
        [0.05, 0.10, 0.80, 0.05],
        [0.05, 0.05, 0.10, 0.80],
        [0.80, 0.05, 0.05, 0.10],
    ], dtype=np.float64)
    # Normalize rows
    pi /= pi.sum(axis=-1, keepdims=True)

    # 2. Generate emission log-likelihoods
    # Distinct states have distinctive signals at different times
    log_likes = np.random.randn(T, K) * 1.5

    # 3. Create a mask with a 10-frame gap (missing observations / occlusions)
    mask = np.ones(T, dtype=bool)
    mask[25:35] = False  # frames 25 through 34 are unobserved

    # 4. Compute exact analytic forward-backward posteriors
    exact_gamma, exact_xi = compute_exact_forward_backward(log_likes, pi, mask)

    # 5. Draw N_samples from Numba FFBS kernel
    N_SAMPLES = 5000
    print(f"Sampling {N_SAMPLES} trajectories from Numba FFBS (T={T}, K={K}, 10 masked frames)...")

    samples = np.zeros((N_SAMPLES, T), dtype=np.int32)
    workspace = np.zeros((T, K), dtype=np.float64)
    rng = np.random.default_rng(123)
    pi_T = np.ascontiguousarray(pi.T)

    for s in range(N_SAMPLES):
        uniforms = rng.uniform(0.0, 1.0, size=T)
        z_s = np.zeros(T, dtype=np.int32)
        _numba_ffbs_single_sequence(log_likes, pi_T, mask, uniforms, z_s, workspace)
        samples[s] = z_s

    # 6. Empirical marginals P(z_t = k)
    emp_gamma = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        counts = np.bincount(samples[:, t], minlength=K)
        emp_gamma[t] = counts / N_SAMPLES

    # Max Absolute Error across all times and states
    mae_marginals = np.max(np.abs(emp_gamma - exact_gamma))
    corr_marginals = np.corrcoef(emp_gamma.flatten(), exact_gamma.flatten())[0, 1]

    print("\n[Test 1: Temporal Marginals P(z_t = k | y)]")
    print(f"  * Correlation with exact FB: r = {corr_marginals:.5f}")
    print(f"  * Max Absolute Error:        MAE = {mae_marginals:.4f} (Monte Carlo 99% bound ~ 0.025)")
    assert corr_marginals > 0.99, f"Marginal correlation too low: {corr_marginals}"
    assert mae_marginals < 0.03, f"Marginal MAE too high: {mae_marginals}"
    print("  --> PASSED (Temporal marginals match exact analytic forward-backward)")

    # 7. Empirical Pairwise Transitions P(z_t = j, z_{t-1} = i)
    emp_xi = np.zeros((T - 1, K, K), dtype=np.float64)
    for t in range(T - 1):
        for s in range(N_SAMPLES):
            i = samples[s, t]
            j = samples[s, t + 1]
            emp_xi[t, i, j] += 1.0
        emp_xi[t] /= N_SAMPLES

    corr_transitions = np.corrcoef(emp_xi.flatten(), exact_xi.flatten())[0, 1]
    mae_transitions = np.max(np.abs(emp_xi - exact_xi))

    print("\n[Test 2: Pairwise Transition Structure P(z_{t-1}=i, z_t=j | y)]")
    print(f"  * Correlation with exact FB: r = {corr_transitions:.5f}")
    print(f"  * Max Absolute Error:        MAE = {mae_transitions:.4f}")
    assert corr_transitions > 0.99, f"Transition correlation too low: {corr_transitions}"
    assert mae_transitions < 0.03, f"Transition MAE too high: {mae_transitions}"
    print("  --> PASSED (Sequential transition structure matches exact temporal dynamics)")

    # 8. Masked Frames Gap Analysis
    # Across masked frames 25:35, verify that state transitions obey the transition matrix
    gap_samples = samples[:, 25:35]
    print("\n[Test 3: Occlusion / Masked Frames Propagation (Frames 25:35)]")
    gap_corr = np.corrcoef(emp_gamma[25:35].flatten(), exact_gamma[25:35].flatten())[0, 1]
    print(f"  * Masked frames distribution correlation: r = {gap_corr:.5f}")
    assert gap_corr > 0.99, f"Masked frames correlation too low: {gap_corr}"
    print("  --> PASSED (Transition matrix cleanly propagates through unobserved frames)")

    print("\n" + "=" * 65)
    print("  ALL HMM TEMPORAL POSTERIOR & SEQUENCE TESTS PASSED (r > 0.99)!")
    print("=" * 65)


if __name__ == "__main__":
    run_temporal_posterior_test()
