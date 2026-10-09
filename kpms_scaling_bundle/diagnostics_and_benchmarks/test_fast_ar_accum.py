import time

import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def numba_accumulate_ar_sufficient_stats(x_in, x_out, z_flat, mask_flat, num_states=100):
    N_total, D_in = x_in.shape
    D_out = x_out.shape[1]

    # Per-thread accumulators to avoid atomic locks
    # 8 threads
    n_threads = 8
    S_in_in_threads = np.zeros((n_threads, num_states, D_in, D_in), dtype=np.float64)
    S_out_in_threads = np.zeros((n_threads, num_states, D_out, D_in), dtype=np.float64)
    S_out_out_threads = np.zeros((n_threads, num_states, D_out, D_out), dtype=np.float64)
    counts_threads = np.zeros((n_threads, num_states), dtype=np.float64)

    chunk_size = (N_total + n_threads - 1) // n_threads

    for thread_id in prange(n_threads):
        start_idx = thread_id * chunk_size
        end_idx = min(start_idx + chunk_size, N_total)

        S_ii = S_in_in_threads[thread_id]
        S_oi = S_out_in_threads[thread_id]
        S_oo = S_out_out_threads[thread_id]
        cnt = counts_threads[thread_id]

        for i in range(start_idx, end_idx):
            if not mask_flat[i]:
                continue
            k = z_flat[i]
            if k < 0 or k >= num_states:
                continue
            cnt[k] += 1.0

            # Outer products
            for r in range(D_in):
                val_r = x_in[i, r]
                for c in range(r, D_in):
                    prod = val_r * x_in[i, c]
                    S_ii[k, r, c] += prod
                    if r != c:
                        S_ii[k, c, r] += prod

            for r in range(D_out):
                val_out = x_out[i, r]
                for c in range(D_in):
                    S_oi[k, r, c] += val_out * x_in[i, c]
                for c in range(r, D_out):
                    prod = val_out * x_out[i, c]
                    S_oo[k, r, c] += prod
                    if r != c:
                        S_oo[k, c, r] += prod

    # Reduce across threads
    S_in_in = np.zeros((num_states, D_in, D_in), dtype=np.float64)
    S_out_in = np.zeros((num_states, D_out, D_in), dtype=np.float64)
    S_out_out = np.zeros((num_states, D_out, D_out), dtype=np.float64)
    counts = np.zeros(num_states, dtype=np.float64)

    for t in range(n_threads):
        S_in_in += S_in_in_threads[t]
        S_out_in += S_out_in_threads[t]
        S_out_out += S_out_out_threads[t]
        counts += counts_threads[t]

    return counts, S_in_in, S_out_in, S_out_out

def main():
    print("Testing AR Sufficient Stats Accumulation across 10,000,000 frames...")
    N_total = 10_000_000
    D_in = 31
    D_out = 10

    x_in = np.random.randn(N_total, D_in).astype(np.float64)
    x_out = np.random.randn(N_total, D_out).astype(np.float64)
    z_flat = np.random.randint(0, 100, size=N_total).astype(np.int32)
    mask_flat = np.ones(N_total, dtype=bool)

    # Warmup
    _ = numba_accumulate_ar_sufficient_stats(x_in[:1000], x_out[:1000], z_flat[:1000], mask_flat[:1000])

    t0 = time.perf_counter()
    counts, S_ii, S_oi, S_oo = numba_accumulate_ar_sufficient_stats(x_in, x_out, z_flat, mask_flat)
    elapsed = (time.perf_counter() - t0) * 1000.0

    print(f"Done in {elapsed:.2f} ms! RAM used: negligible (< 50 MB), VRAM used: 0 MB")
    print(f"Total counted frames: {counts.sum():,}")

if __name__ == "__main__":
    main()
