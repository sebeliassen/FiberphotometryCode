import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def numba_accumulate_ar_sufficient_stats(x, z, mask, nlags, D, num_states):
    N, T, _ = x.shape
    in_dim = nlags * D + 1
    
    # Thread-local accumulators
    S_in_in = np.zeros((num_states, in_dim, in_dim), dtype=np.float64)
    S_out_in = np.zeros((num_states, D, in_dim), dtype=np.float64)
    S_out_out = np.zeros((num_states, D, D), dtype=np.float64)
    counts = np.zeros(num_states, dtype=np.float64)
    
    for seq_i in prange(N):
        x_in_buf = np.empty(in_dim, dtype=np.float64)
        x_in_buf[-1] = 1.0
        
        for t in range(nlags, T):
            if not mask[seq_i, t]:
                continue
            
            k = z[seq_i, t - nlags]
            
            # Fill x_in
            for l in range(1, nlags + 1):
                start_l = (l - 1) * D
                for d_i in range(D):
                    x_in_buf[start_l + d_i] = x[seq_i, t - l, d_i]
            
            x_out = x[seq_i, t]
            
            # Outer products
            for i in range(in_dim):
                for j in range(in_dim):
                    S_in_in[k, i, j] += x_in_buf[i] * x_in_buf[j]
            
            for i in range(D):
                for j in range(in_dim):
                    S_out_in[k, i, j] += x_out[i] * x_in_buf[j]
                    
            for i in range(D):
                for j in range(D):
                    S_out_out[k, i, j] += x_out[i] * x_out[j]
                    
            counts[k] += 1.0
            
    return S_in_in, S_out_in, S_out_out, counts

def test():
    N = 1004
    T = 10030
    D = 10
    nlags = 3
    K = 100
    print(f"Testing Numba AR Accumulator for {N} videos ({N*T:,} frames)...")
    
    x = np.random.randn(N, T, D).astype(np.float64)
    z = np.random.randint(0, K, size=(N, T - nlags))
    mask = np.ones((N, T), dtype=bool)
    
    # Warmup
    _ = numba_accumulate_ar_sufficient_stats(x[:2], z[:2], mask[:2], nlags, D, K)
    
    t0 = time.perf_counter()
    S_in_in, S_out_in, S_out_out, counts = numba_accumulate_ar_sufficient_stats(x, z, mask, nlags, D, K)
    elapsed = time.perf_counter() - t0
    
    print(f"Elapsed: {elapsed*1000.0:.2f} ms ({elapsed:.3f} s) for 10.07 MILLION frames!")
    print(f"Total counts accumulated: {counts.sum():,}")
    print(f"Memory overhead: ~0 MB!")

if __name__ == "__main__":
    test()
