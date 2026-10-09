#!/usr/bin/env python3
"""
Test script for verified optimization of:
1. Streaming HMM with contiguous pi_T
2. Preallocated workspace for continuous Banded Cholesky
3. LAPACK dpbtrf + BLAS dtbsv solver parity
"""
import sys, os
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
from numba import njit, prange
from scipy.linalg.lapack import dpbtrf
from scipy.linalg.blas import dtbsv
import concurrent.futures
import time

import utils.resample_batching as rb

def test_ffbs_parity():
    print("--- Testing FFBS Contiguous pi_T Parity ---")
    T, K = 1000, 100
    rng = np.random.default_rng(42)
    log_likes = rng.standard_normal((T, K))
    pi = rng.uniform(0.1, 1.0, (K, K))
    pi /= pi.sum(axis=1, keepdims=True)
    pi_T = np.ascontiguousarray(pi.T)
    mask = np.ones(T, dtype=bool)
    mask[100:150] = False
    uniforms = rng.uniform(0, 1, T)

    z1 = np.zeros(T, dtype=np.int32)
    P1 = np.zeros((T, K), dtype=np.float64)
    rb._numba_ffbs_single_sequence(log_likes, pi, mask, uniforms, z1, P1)

    # Contiguous version
    @njit(fastmath=True)
    def ffbs_contiguous(log_likes, pi_T, mask, uniforms, z_out, P_workspace):
        T, K = log_likes.shape
        max_ll = -1e300
        for k in range(K):
            if log_likes[0, k] > max_ll: max_ll = log_likes[0, k]
        sum_p = 0.0
        for k in range(K):
            val = np.exp(log_likes[0, k] - max_ll) if mask[0] else 1.0
            P_workspace[0, k] = val
            sum_p += val
        inv_sum = 1.0 / (sum_p + 1e-300)
        for k in range(K): P_workspace[0, k] *= inv_sum

        pred = np.empty(K, dtype=np.float64)
        for t in range(1, T):
            p_prev = P_workspace[t - 1]
            for k in range(K):
                s = 0.0
                for j in range(K):
                    s += p_prev[j] * pi_T[k, j]
                pred[k] = s
            if mask[t]:
                max_ll = -1e300
                for k in range(K):
                    if log_likes[t, k] > max_ll: max_ll = log_likes[t, k]
                sum_p = 0.0
                for k in range(K):
                    val = pred[k] * np.exp(log_likes[t, k] - max_ll)
                    P_workspace[t, k] = val
                    sum_p += val
            else:
                sum_p = 0.0
                for k in range(K):
                    val = pred[k]
                    P_workspace[t, k] = val
                    sum_p += val
            inv_sum = 1.0 / (sum_p + 1e-300)
            for k in range(K): P_workspace[t, k] *= inv_sum

        z_out[T - 1] = rb._numba_sample_categorical(P_workspace[T - 1], uniforms[T - 1])
        q = np.empty(K, dtype=np.float64)
        for t in range(T - 2, -1, -1):
            next_s = z_out[t + 1]
            sum_q = 0.0
            p_curr = P_workspace[t]
            for j in range(K):
                val = p_curr[j] * pi_T[next_s, j]
                q[j] = val
                sum_q += val
            inv_sum_q = 1.0 / (sum_q + 1e-300)
            for j in range(K): q[j] *= inv_sum_q
            z_out[t] = rb._numba_sample_categorical(q, uniforms[t])

    z2 = np.zeros(T, dtype=np.int32)
    P2 = np.zeros((T, K), dtype=np.float64)
    ffbs_contiguous(log_likes, pi_T, mask, uniforms, z2, P2)

    assert np.array_equal(z1, z2), "State sequences do not match!"
    print("PASSED: FFBS contiguous layout matches 100% sample-for-sample.")

def test_lapack_banded_parity():
    print("--- Testing LAPACK Banded Cholesky & BLAS Solve Parity ---")
    N = 100300
    bandwidth = 39
    rng = np.random.default_rng(123)
    ab = rng.standard_normal((bandwidth + 1, N)) * 0.01
    ab[0] = 10.0
    g = rng.standard_normal(N)
    eta = rng.standard_normal(N)

    # Numba
    ab_n = ab.copy()
    rb._numba_banded_cholesky_inplace(ab_n, bandwidth, N)
    w_n = np.zeros(N)
    rb._numba_banded_forward_solve(ab_n, bandwidth, N, g, w_n)
    w_n += eta
    x_n = np.zeros(N)
    rb._numba_banded_backward_solve(ab_n, bandwidth, N, w_n, x_n)

    # LAPACK + BLAS
    ab_l = ab.copy()
    c_band, info = dpbtrf(ab_l, lower=1, overwrite_ab=1)
    assert info == 0, "dpbtrf failed!"
    w_l = dtbsv(bandwidth, c_band, g, lower=1, trans=0, overwrite_x=0)
    w_l += eta
    x_l = dtbsv(bandwidth, c_band, w_l, lower=1, trans=1, overwrite_x=0)

    diff = np.max(np.abs(x_n - x_l))
    print(f"Max absolute difference: {diff:.6e}")
    assert diff < 1e-12, "LAPACK solve mismatch!"
    print("PASSED: LAPACK dpbtrf + BLAS dtbsv matches Numba to machine precision.")

if __name__ == "__main__":
    test_ffbs_parity()
    test_lapack_banded_parity()
    print("\nALL STANDALONE PARITY CHECKS PASSED!")
