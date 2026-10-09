import time
import argparse
import sys
import os
import gc
import jax
import jax.numpy as jnp
import keypoint_moseq as kpms

# Add local directory to path to find utils
sys.path.append(os.getcwd())

# Import the setup module we just created
from generate_synthetic_env import generate_synthetic_environment

# Import your optimized code
try:
    from utils.batch_init import batched_init_model_by_video
    print("Loaded optimized init from: utils.batch_init")
except ImportError:
    try:
        from utils.resample_batching import batched_init_model_by_video
        print("Loaded optimized init from: utils.resample_batching")
    except ImportError:
        print("WARNING: Could not load optimized init function!")

def get_peak_mem_gb():
    """Returns peak GPU memory usage in GB."""
    jax.block_until_ready(jnp.array(0))
    gc.collect()
    try:
        stats = jax.devices()[0].memory_stats()
        return stats['peak_bytes_in_use'] / 1e9
    except:
        return 0.0

def benchmark(mode):
    # 1. Prepare Environment (Huge Scale)
    # 200 videos * 50,000 frames = 10 Million frames
    data, pca, config = generate_synthetic_environment(
        N_seqs=400, 
        Frames=10000, 
        Latent=10
    )
    
    jax.clear_caches()
    baseline_mem = get_peak_mem_gb()
    print(f"--- BENCHMARK: {mode.upper()} ---")
    print(f"Baseline Memory: {baseline_mem:.2f} GB")
    
    start_time = time.time()
    
    try:
        if mode == 'standard':
            print("Running kpms.init_model (Standard)...")
            model = kpms.init_model(data, pca=pca, **config)
            # Access variable to force computation
            jax.block_until_ready(model['states']['x'])
            
        elif mode == 'batched':
            print("Running batched_init_model_by_video (Optimized)...")
            model = batched_init_model_by_video(
                data, pca, config,
                params_subset_size=5, 
                batch_size=5
            )
            
        duration = time.time() - start_time
        peak_mem = get_peak_mem_gb()
        
        print(f"\nSUCCESS")
        print(f"Time: {duration:.2f}s")
        print(f"Peak Mem: {peak_mem:.2f} GB")
        
    except Exception as e:
        print(f"\nCRASHED: {e}")
        print(f"Peak Mem before crash: {get_peak_mem_gb():.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['standard', 'batched'], required=True)
    args = parser.parse_args()
    
    benchmark(args.mode)