import os
import sys
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import jax
import jax.numpy as jnp
import jax.random as jr
from numba import njit

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
jax.config.update("jax_enable_x64", True)

def run_advanced_frontiers_test():
    print("=" * 85)
    print("       ADVANCED INFERENCE FRONTIERS FOR KPMS CONTINUOUS LATENTS (x)")
    print("=" * 85)

    T = 10030
    D = 10      # latent dimension
    nlags = 3
    D_state = D * nlags  # 30
    D_obs = 26  # keypoint observations

    np.random.seed(42)
    key = jr.PRNGKey(42)

    # Synthetic realistic SLDS parameters
    A = np.eye(D_state) * 0.95
    A[:D, :D] = 0.8 * np.eye(D)
    Q = np.eye(D_state) * 0.1
    Q_inv = np.linalg.inv(Q)
    H = np.zeros((D_obs, D_state))
    H[:, :D] = np.random.randn(D_obs, D) * 0.2
    R_diag = np.ones(D_obs) * 0.15
    R_inv_diag = 1.0 / R_diag

    y = np.random.randn(T, D_obs)

    # -------------------------------------------------------------
    # Frontier 1: Precision-Based Banded GMRF Sampling
    # -------------------------------------------------------------
    print("\n1. Testing Precision-Based Banded Cholesky Sampling (GMRF formulation):")
    print("-" * 85)

    n_total = T * D
    bandwidth = nlags * D  # 30

    print(f"   * Total GMRF Dimension: {n_total:,} variables")
    print(f"   * Bandwidth (diagonals): {bandwidth}")

    # Build banded precision matrix (lower=True format: main diagonal at row 0)
    ab = np.zeros((bandwidth + 1, n_total))
    ab[0, :] = 10.0 + np.random.uniform(0.1, 0.5, size=n_total)
    for b in range(1, bandwidth + 1):
        ab[b, :-b] = np.random.uniform(-0.05, 0.05, size=n_total - b)

    # Benchmark Banded Cholesky Factorization
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        c = sla.cholesky_banded(ab, lower=True, overwrite_ab=False)
    t_chol_banded = ((time.perf_counter() - t0) / iters) * 1000.0

    # Benchmark Banded Triangular Solve (Sampling step: L^T x = z)
    z_noise = np.random.randn(n_total)
    t0 = time.perf_counter()
    for _ in range(iters):
        x_sample = sla.cho_solve_banded((c, True), z_noise)
    t_solve_banded = ((time.perf_counter() - t0) / iters) * 1000.0

    t_total_banded = t_chol_banded + t_solve_banded
    print(f"   * Scipy Banded Cholesky Factorization : {t_chol_banded:7.2f} ms")
    print(f"   * Scipy Banded Triangular Solve       : {t_solve_banded:7.2f} ms")
    print(f"   * Total GMRF Banded Sample Time       : {t_total_banded:7.2f} ms (vs ~4,140 ms in Kalman scan!)")
    print(f"   * Speedup potential                  : {4140.0 / t_total_banded:7.2f}x")

    # -------------------------------------------------------------
    # Frontier 2: Numba CPU Ultra-Optimized Kalman Filter & Smoother
    # -------------------------------------------------------------
    print("\n2. Testing Numba C-Compiled CPU Sequential Kalman Filter & Backward Sampler:")
    print("-" * 85)

    @njit(fastmath=True)
    def numba_kalman_sampler(y, A, Q, H, R_diag, T, D_state, D_obs):
        filt_m = np.zeros((T, D_state))
        filt_P = np.zeros((T, D_state, D_state))
        
        m = np.zeros(D_state)
        P = np.eye(D_state) * 10.0
        
        H_T_Rinv = np.zeros((D_state, D_obs))
        for i in range(D_state):
            for j in range(D_obs):
                H_T_Rinv[i, j] = H[j, i] / R_diag[j]
        
        H_T_Rinv_H = H_T_Rinv @ H

        # Forward filter
        for t in range(T):
            P_inv = np.linalg.inv(P) + H_T_Rinv_H
            P_filt = np.linalg.inv(P_inv)
            
            y_t = y[t]
            residual = y_t - H @ m
            K = P_filt @ H_T_Rinv
            m_filt = m + K @ residual
            
            filt_m[t] = m_filt
            filt_P[t] = P_filt
            
            # Predict
            m = A @ m_filt
            P = A @ P_filt @ A.T + Q

        # Backward sample
        samples = np.zeros((T, D_state))
        last_L = np.linalg.cholesky(filt_P[T - 1] + np.eye(D_state) * 1e-3)
        curr_x = filt_m[T - 1] + last_L @ np.random.randn(D_state)
        samples[T - 1] = curr_x

        for t in range(T - 2, -1, -1):
            m_t = filt_m[t]
            P_t = filt_P[t]
            pred_cov = A @ P_t @ A.T + Q
            J = P_t @ A.T @ np.linalg.inv(pred_cov)
            smooth_m = m_t + J @ (curr_x - A @ m_t)
            smooth_P = P_t - J @ pred_cov @ J.T + np.eye(D_state) * 1e-3
            L = np.linalg.cholesky(smooth_P)
            curr_x = smooth_m + L @ np.random.randn(D_state)
            samples[t] = curr_x

        return samples

    # Warmup Numba JIT
    _ = numba_kalman_sampler(y, A, Q, H, R_diag, 50, D_state, D_obs)

    # Benchmark Numba on full T=10,030
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = numba_kalman_sampler(y, A, Q, H, R_diag, T, D_state, D_obs)
    t_numba = ((time.perf_counter() - t0) / iters) * 1000.0

    print(f"   * Native C/Numba CPU Kalman Sampler   : {t_numba:7.2f} ms (vs ~4,140 ms GPU JAX lax.scan)")
    print(f"   * Speedup potential                  : {4140.0 / t_numba:7.2f}x")

    # -------------------------------------------------------------
    # Frontier 3: Steady-State Kalman Covariance Convergence Analysis
    # -------------------------------------------------------------
    print("\n3. Testing Steady-State Covariance Convergence |P_t - P_{t-1}| in KPMS:")
    print("-" * 85)
    
    P_curr = np.eye(D_state) * 10.0
    H_T_Rinv = (H.T / R_diag[None, :])
    H_T_Rinv_H = H_T_Rinv @ H
    
    diffs = []
    for step in range(500):
        P_inv = np.linalg.inv(P_curr) + H_T_Rinv_H
        P_filt = np.linalg.inv(P_inv)
        P_next = A @ P_filt @ A.T + Q
        diff = np.linalg.norm(P_next - P_curr, ord='fro')
        diffs.append(diff)
        P_curr = P_next

    print(f"   * Step   1 Diff ||P_1 - P_0||_F       : {diffs[0]:.2e}")
    print(f"   * Step  10 Diff ||P_10 - P_9||_F      : {diffs[9]:.2e}")
    print(f"   * Step  50 Diff ||P_50 - P_49||_F     : {diffs[49]:.2e}")
    print(f"   * Step 100 Diff ||P_100 - P_99||_F    : {diffs[99]:.2e}")
    print(f"   * Step 200 Diff ||P_200 - P_199||_F   : {diffs[199]:.2e}")

    print("=" * 85)

if __name__ == "__main__":
    run_advanced_frontiers_test()
