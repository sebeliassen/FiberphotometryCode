import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.models.keypoint_slds import gibbs, slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error


# =============================================================================
# 1. THE JIT KERNEL (Compiles once per shape)
# =============================================================================
@jax.jit
def _resample_scales_kernel(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    """
    Standard SLDS noise resampling logic, JIT-compiled for speed.
    """
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

# =============================================================================
# 2. THE BATCHED ORCHESTRATOR (Python Loop)
# =============================================================================
def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    """
    A memory-efficient replacement for gibbs.resample_scales.
    It processes data in chunks (Batch Size 50) to prevent OOM on large datasets.
    """
    BATCH_SIZE = 50  # Tuned for RTX 4090 (24GB VRAM)
    N = Y.shape[0]
    results = []

    # Split seeds so every video gets a unique random seed
    seeds = jr.split(seed, N)

    # Loop: Process in chunks
    # Iterations 1-N: Uses cached kernel for shape 50
    # Remainder:      Uses cached kernel for shape (N % 50)
    for i in range(0, N, BATCH_SIZE):
        end = min(i + BATCH_SIZE, N)

        # Slicing creates views (no extra memory)
        s_batch = _resample_scales_kernel(
            seeds[i:end],
            Y[i:end],
            x[i:end],
            v[i:end],
            h[i:end],
            Cd, sigmasq, nu_s, s_0
        )

        # Block to keep GPU memory clean
        s_batch.block_until_ready()
        results.append(s_batch)

    return jnp.concatenate(results, axis=0)

# =============================================================================
# 3. THE ACTIVATOR
# =============================================================================
def apply_optimizations():
    """
    Apply monkey patches to the running Keypoint-MoSeq library.
    Call this once at the start of your pipeline.
    """
    print("--- Applying Custom Kernels ---")

    # 1. Patch Resample Scales (Fixes OOM on 5.6M frames)
    gibbs.resample_scales = batched_resample_scales
    print(f"[Kernel] Optimized 'resample_scales' with Batched Strategy (Batch={50})")

    print("-------------------------------")
