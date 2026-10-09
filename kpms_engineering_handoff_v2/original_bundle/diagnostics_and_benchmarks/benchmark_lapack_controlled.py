#!/usr/bin/env python3
"""
diagnostics/benchmark_lapack_controlled.py

Controlled comparison of Numba Banded Cholesky vs SciPy LAPACK dpbtrf/dtbsv:
1. Controlled BLAS threads (limits=1, user_api='blas') via threadpoolctl.
2. Verified memory layouts (Fortran-contiguous for LAPACK, C-contiguous for Numba).
3. Fresh input matrices for each repeat (preventing biased in-place overwrites).
4. Numerical parity verification (max absolute difference < 1e-12, info == 0).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import scipy.linalg.lapack as lapack
import scipy.linalg.blas as blas
from threadpoolctl import threadpool_limits, threadpool_info
import time
from numba import njit, prange
import utils.resample_batching as rb

@njit(parallel=True, fastmath=True)
def run_numba_parallel(abs_all, gs, etas, xs, bw, n):
    N_s = abs_all.shape[0]
    for i in prange(N_s):
        ab = abs_all[i]
        g = gs[i]
        eta = etas[i]
        x = xs[i]
        rb._numba_banded_cholesky_inplace(ab, bw, n)
        rb._numba_banded_forward_solve_inplace(ab, bw, n, g)
        for j in range(n):
            g[j] += eta[j]
        rb._numba_banded_backward_solve(ab, bw, n, g, x)

def run_benchmark():
    print("=" * 70)
    print("  CONTROLLED LAPACK vs NUMBA BANDED CHOLESKY BENCHMARK")
    print("=" * 70, flush=True)

    print("\n[Active Threadpool Configuration]")
    for info in threadpool_info():
        print(f"  - {info['user_api']}: {info['internal_api']} ({info['num_threads']} threads, {info['filepath']})", flush=True)

    N = 100300
    bw = 39
    num_seqs = 16
    rng = np.random.default_rng(12345)

    def make_fresh_data(count):
        ab_c = rng.standard_normal((count, bw + 1, N)) * 0.01
        for i in range(count):
            ab_c[i, 0] = 10.0
        ab_c = np.ascontiguousarray(ab_c)
        ab_f = [np.asfortranarray(ab_c[i].copy()) for i in range(count)]
        for arr in ab_f:
            assert arr.flags['F_CONTIGUOUS'], "Must be Fortran contiguous for LAPACK"
        g = rng.standard_normal((count, N))
        eta = rng.standard_normal((count, N))
        return ab_c, ab_f, g, eta

    # 1. Single Sequence Latency Test (Serial execution, BLAS=1)
    print("\n--- 1. Single-Sequence Latency (Serial Execution) ---", flush=True)
    ab_c, ab_f, g, eta = make_fresh_data(1)

    with threadpool_limits(limits=1, user_api='blas'):
        # Numba serial
        ab_n = ab_c[0].copy()
        g_n = g[0].copy()
        eta_n = eta[0].copy()
        x_n = np.zeros(N)

        t0 = time.perf_counter()
        rb._numba_banded_cholesky_inplace(ab_n, bw, N)
        rb._numba_banded_forward_solve_inplace(ab_n, bw, N, g_n)
        for j in range(N):
            g_n[j] += eta_n[j]
        rb._numba_banded_backward_solve(ab_n, bw, N, g_n, x_n)
        t_numba_single = time.perf_counter() - t0

        # LAPACK serial (Fortran layout)
        ab_l = ab_f[0].copy()
        g_l = g[0].copy()
        eta_l = eta[0].copy()

        t0 = time.perf_counter()
        c_band, info = lapack.dpbtrf(ab_l, lower=1, overwrite_ab=1)
        assert info == 0, f"dpbtrf failed with info={info}"
        w = blas.dtbsv(bw, c_band, g_l, lower=1, trans=0, overwrite_x=1)
        w += eta_l
        x_l = blas.dtbsv(bw, c_band, w, lower=1, trans=1, overwrite_x=1)
        t_lapack_single = time.perf_counter() - t0

    max_diff = np.max(np.abs(x_n - x_l))
    print(f"  * Numba single sequence:  {t_numba_single*1000:.2f} ms")
    print(f"  * LAPACK single sequence: {t_lapack_single*1000:.2f} ms (BLAS=1, Fortran layout)")
    print(f"  * Numerical Max Diff:     {max_diff:.3e} (info={info})")
    assert max_diff < 1e-12, "Numerical parity failed!"
    print("  --> Exact numerical parity confirmed (< 1e-12).", flush=True)

    # 2. Multi-Sequence Throughput (16 Sequences across 32 Cores)
    print("\n--- 2. Multi-Sequence Throughput (16 Sequences across 32 Cores) ---", flush=True)

    # Warmup Numba parallel solver
    ab_c_warm, _, g_warm, eta_warm = make_fresh_data(2)
    x_warm = np.zeros((2, N))
    run_numba_parallel(ab_c_warm, g_warm, eta_warm, x_warm, bw, N)

    # Numba Parallel Benchmark (3 repeats with fresh data)
    times_numba = []
    for _ in range(3):
        ab_c_fresh, _, g_fresh, eta_fresh = make_fresh_data(num_seqs)
        x_out = np.zeros((num_seqs, N))
        t0 = time.perf_counter()
        run_numba_parallel(ab_c_fresh, g_fresh, eta_fresh, x_out, bw, N)
        times_numba.append(time.perf_counter() - t0)
    avg_numba = np.mean(times_numba)
    print(f"  * Native Numba Parallel:  {avg_numba:.4f} s total ({avg_numba/num_seqs*1000:.2f} ms/seq amortized)", flush=True)

    # Serial LAPACK loop benchmark (3 repeats with fresh data, BLAS=1)
    times_lapack_serial = []
    with threadpool_limits(limits=1, user_api='blas'):
        for _ in range(3):
            _, ab_f_fresh, g_fresh, eta_fresh = make_fresh_data(num_seqs)
            t0 = time.perf_counter()
            for i in range(num_seqs):
                c_band, info = lapack.dpbtrf(ab_f_fresh[i], lower=1, overwrite_ab=1)
                w = blas.dtbsv(bw, c_band, g_fresh[i], lower=1, trans=0, overwrite_x=1)
                w += eta_fresh[i]
                blas.dtbsv(bw, c_band, w, lower=1, trans=1, overwrite_x=1)
            times_lapack_serial.append(time.perf_counter() - t0)
    avg_lapack_serial = np.mean(times_lapack_serial)
    print(f"  * Serial LAPACK (BLAS=1): {avg_lapack_serial:.4f} s total ({avg_lapack_serial/num_seqs*1000:.2f} ms/seq amortized)", flush=True)

    print("\n[Controlled Summary]")
    print(f"  1. Single sequence latency: Numba = {t_numba_single*1000:.2f} ms vs LAPACK = {t_lapack_single*1000:.2f} ms")
    print(f"  2. Multi-sequence throughput: Numba Parallel = {avg_numba:.4f} s vs Serial LAPACK = {avg_lapack_serial:.4f} s")
    print(f"  3. Numerical Agreement: Max absolute difference = {max_diff:.3e}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_benchmark()
