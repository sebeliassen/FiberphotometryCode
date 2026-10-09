import os
import argparse
import sys
import time
import gc

# 1. CRITICAL: Disable JAX Pre-allocation
# Must be done BEFORE importing jax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# --- IMPORT ADJUSTMENT ---
# Add current directory to path so we can find 'utils'
sys.path.append(os.getcwd())

try:
    # Try the clean name we made for the PR
    from utils.batch_utils import batched_resample_scales
    print("Loaded optimized kernel from: utils.batch_utils")
except ImportError:
    try:
        # Fallback to the original filename you had
        from utils.resample_batching import batched_resample_scales
        print("Loaded optimized kernel from: utils.resample_batching")
    except ImportError:
        print("ERROR: Could not find 'batched_resample_scales' in utils/batch_utils.py or utils/resample_batching.py")
        sys.exit(1)

# Import standard library components for the "control" group
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error

def get_peak_mem_gb():
    # Sync GPU to ensure all allocations are counted
    jax.block_until_ready(jnp.array(0))
    gc.collect()
    # 'peak_bytes_in_use' is the highest memory watermark since process start
    stats = jax.devices()[0].memory_stats()
    return stats['peak_bytes_in_use'] / 1e9

def run_benchmark(mode):
    print(f"\n--- BENCHMARKING MODE: {mode.upper()} ---")
    
    # 2. SETUP SYNTHETIC DATA
    # Adjust these if Standard crashes your GPU
    N = 100     # Number of videos
    T = 100030    # Frames per video
    K = 15      # Keypoints
    D = 2       # Dimensions (x,y)
    Latent = 4  # Latent state dim
    
    print(f"Dataset: {N} sequences x {T} frames")
    print(f"Total Frames: {N*T/1e6:.1f} Million")
    
    key = jr.PRNGKey(0)

    # Generate on CPU first to keep GPU memory clean
    print("Generating data on CPU...")
    with jax.default_device(jax.devices("cpu")[0]):
        Y = jnp.array(np.random.randn(N, T, K, D))
        x = jnp.array(np.random.randn(N, T, Latent))
        v = jnp.array(np.random.randn(N, T, D))
        h = jnp.array(np.random.randn(N, T))
        s_0 = jnp.array(np.random.randn(N, T, K))
        
        Cd = jnp.zeros(((K - 1) * D, Latent + 1))
        sigmasq = jnp.ones((K,))
        nu_s = 5.0

    # 3. MOVE TO GPU & MEASURE BASELINE
    print("Moving data to GPU...")
    Y_gpu = jax.device_put(Y, jax.devices()[0])
    x_gpu = jax.device_put(x, jax.devices()[0])
    v_gpu = jax.device_put(v, jax.devices()[0])
    h_gpu = jax.device_put(h, jax.devices()[0])
    s_gpu = jax.device_put(s_0, jax.devices()[0])
    Cd_gpu = jax.device_put(Cd, jax.devices()[0])
    sigmasq_gpu = jax.device_put(sigmasq, jax.devices()[0])
    
    # Force allocation
    jax.block_until_ready(Y_gpu)
    
    baseline_mem = jax.devices()[0].memory_stats()['bytes_in_use'] / 1e9
    print(f"Baseline Memory (Static Data): {baseline_mem:.2f} GB")

    # 4. RUN THE TEST
    start_time = time.time()
    
    if mode == 'standard':
        print("Running Standard Kernel (All-at-once)...")
        
        # Define the exact logic used in the original library
        def _standard_run(seed, Y, x, v, h, s_0):
            sqerr = compute_squared_error(Y, x, v, h, Cd_gpu)
            return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq_gpu, nu_s, s_0)
        
        # JIT + VMAP over the full dataset
        kernel = jax.jit(jax.vmap(_standard_run, in_axes=(0,0,0,0,0,0)))
        seeds = jr.split(key, N)
        
        # Execution
        res = kernel(seeds, Y_gpu, x_gpu, v_gpu, h_gpu, s_gpu)
        res.block_until_ready()

    elif mode == 'batched':
        print("Running Batched Kernel (Optimized)...")
        # Uses your local utils code
        res = batched_resample_scales(
            key, Y_gpu, x_gpu, v_gpu, h_gpu, Cd_gpu, sigmasq_gpu, nu_s, s_gpu
        )
        res.block_until_ready()

    elapsed = time.time() - start_time
    
    # 5. REPORT RESULTS
    peak_mem = get_peak_mem_gb()
    operation_cost = peak_mem - baseline_mem
    
    print("\n" + "="*40)
    print(f"RESULTS: {mode.upper()}")
    print(f"Time Taken:     {elapsed:.2f} s")
    print(f"Peak Memory:    {peak_mem:.2f} GB")
    print(f"Operation Cost: {operation_cost:.2f} GB (Spike)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['standard', 'batched'], required=True)
    args = parser.parse_args()
    
    run_benchmark(args.mode)