import gc
import os
import sys

import numpy as np
import psutil

np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers/keypoint_moseq_syll")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, KPMS_DIR)

from utils.resample_batching import _numba_accumulate_ar_stats


def get_ram_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def main():
    print("=" * 80)
    print("      MEMORY TRACE PROFILER ACROSS SCALING VIDEO COUNTS")
    print("=" * 80)
    print(f"Base RAM: {get_ram_gb():.2f} GB\n")

    for N in [50, 200, 500, 1004]:
        T = 10030
        D = 10
        nlags = 3
        D_in = nlags * D + 1
        num_states = 100

        print(f"--- Testing N = {N} videos ({N * T:,} frames) ---")

        # 1. Allocate states
        x = np.random.randn(N, T, D).astype(np.float64)
        z = np.random.randint(0, num_states, size=(N, T - nlags)).astype(np.int32)
        mask = np.ones((N, T), dtype=bool)

        ram_after_alloc = get_ram_gb()
        print(f"  [1] States allocated: RAM = {ram_after_alloc:.2f} GB")

        # 2. Extract lags in-place
        x_in = np.empty((N * (T - nlags), D_in), dtype=np.float64)
        x_out = np.empty((N * (T - nlags), D), dtype=np.float64)

        # Fill in-place without giant intermediate allocations
        row = 0
        for seq_i in range(N):
            for t in range(nlags, T):
                x_out[row, :] = x[seq_i, t]
                x_in[row, 0] = 1.0  # affine bias
                col = 1
                for l in range(1, nlags + 1):
                    x_in[row, col:col+D] = x[seq_i, t - l]
                    col += D
                row += 1

        ram_after_lags = get_ram_gb()
        print(f"  [2] In-place lags built: RAM = {ram_after_lags:.2f} GB")

        # 3. Numba accumulate
        mask_flat = mask[:, nlags:].reshape(-1)
        z_flat = z.reshape(-1)

        counts, S_ii, S_oi, S_oo = _numba_accumulate_ar_stats(x_in, x_out, z_flat, mask_flat, num_states)
        ram_after_accum = get_ram_gb()
        print(f"  [3] Numba accumulated: RAM = {ram_after_accum:.2f} GB")

        # Clean up
        del x, z, mask, x_in, x_out, mask_flat, z_flat, counts, S_ii, S_oi, S_oo
        gc.collect()
        print(f"  [4] After GC: RAM = {get_ram_gb():.2f} GB\n")

if __name__ == "__main__":
    main()
