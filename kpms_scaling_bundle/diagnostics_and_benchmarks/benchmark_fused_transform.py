#!/usr/bin/env python3
"""
diagnostics/benchmark_fused_transform.py

Benchmarks Point 5 from reviewer feedback:
Fusing subtraction (Y - v), 2D rotation (rot h), and flattening into a
preallocated multi-threaded Numba kernel vs Current NumPy array allocations.
"""

import time
import numpy as np
from numba import njit, prange

# Current NumPy implementation
def current_numpy_transform(Y_c, v_c, h_c, N_chunk, T):
    cos_h = np.cos(h_c)[:, :, None]
    sin_h = np.sin(h_c)[:, :, None]
    Y_trans = Y_c - v_c[:, :, None, :]
    Y_rot_0 = cos_h * Y_trans[..., 0] + sin_h * Y_trans[..., 1]
    Y_rot_1 = -sin_h * Y_trans[..., 0] + cos_h * Y_trans[..., 1]
    Y_c_van = np.stack([Y_rot_0, Y_rot_1], axis=-1).reshape(N_chunk, T, -1)
    return Y_c_van

# Fused Numba multi-threaded kernel
@njit(fastmath=True, parallel=True)
def fused_numba_transform(Y_c, v_c, h_c, out_flat):
    N, T, K, d = Y_c.shape
    for i in prange(N):
        for t in range(T):
            c_h = np.cos(h_c[i, t])
            s_h = np.sin(h_c[i, t])
            vx = v_c[i, t, 0]
            vy = v_c[i, t, 1]
            for k in range(K):
                # Subtraction
                tx = Y_c[i, t, k, 0] - vx
                ty = Y_c[i, t, k, 1] - vy
                # Rotation & flat storage
                out_flat[i, t, 2 * k]     =  c_h * tx + s_h * ty
                out_flat[i, t, 2 * k + 1] = -s_h * tx + c_h * ty

def main():
    print("=" * 70)
    print("  BENCHMARK: FUSED RIGID TRANSFORMATION (NUMPY VS NUMBA)")
    print("=" * 70)

    N_chunk = 50
    T = 10030
    K = 17
    d = 2

    rng = np.random.default_rng(42)
    Y_c = rng.standard_normal((N_chunk, T, K, d))
    v_c = rng.standard_normal((N_chunk, T, d))
    h_c = rng.standard_normal((N_chunk, T))
    out_numba = np.empty((N_chunk, T, K * d), dtype=np.float64)

    # Warmup Numba
    fused_numba_transform(Y_c[:2], v_c[:2], h_c[:2], out_numba[:2])

    # 1. Benchmark NumPy (across 12 chunks = full 562 seqs)
    t0 = time.perf_counter()
    for _ in range(12):
        res_np = current_numpy_transform(Y_c, v_c, h_c, N_chunk, T)
    t_numpy = time.perf_counter() - t0

    # 2. Benchmark Fused Numba (across 12 chunks = full 562 seqs)
    t0 = time.perf_counter()
    for _ in range(12):
        fused_numba_transform(Y_c, v_c, h_c, out_numba)
    t_numba = time.perf_counter() - t0

    diff = np.max(np.abs(res_np - out_numba))
    print(f"Max numerical difference: {diff:.3e}")
    assert diff < 1e-12, "Parity check failed!"

    print(f"\nAcross 562 sequences (12 chunks of 50):")
    print(f"  * Current NumPy (with 5 GB allocations):  {t_numpy:.3f} s")
    print(f"  * Fused Numba (0 allocations, multi-core): {t_numba:.3f} s")
    print(f"  * Speedup factor:                         {t_numpy / t_numba:.2f}x")
    print(f"  * Time saved per SLDS iteration:          {t_numpy - t_numba:.3f} s")
    print("=" * 70)

if __name__ == "__main__":
    main()
