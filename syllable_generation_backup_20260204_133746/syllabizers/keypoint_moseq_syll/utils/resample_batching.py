import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error
from tqdm import tqdm

# =============================================================================
# 1. THE JIT KERNEL
# =============================================================================

def _single_sequence_resample(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

_kernel_vmap = jax.vmap(
    _single_sequence_resample, 
    in_axes=(0, 0, 0, 0, 0, None, None, None, 0) 
)

@jax.jit
def _resample_scales_kernel_batched(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    return _kernel_vmap(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0)

# =============================================================================
# 2. THE BATCHED ORCHESTRATOR (Refactored)
# =============================================================================

def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    BATCH_SIZE = 20
    N = Y.shape[0]
    seeds = jr.split(seed, N)
    results = []

    # Check s_0 type once
    is_s0_per_seq = (hasattr(s_0, "shape") and s_0.shape[0] == N)
    
    # Helper for padding
    def pad_arr(arr, target_len):
        pad_size = target_len - arr.shape[0]
        padding = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
        return jnp.pad(arr, padding, mode='edge')

    # Group standard inputs to slice them easily
    # Note: s_0 is handled separately due to broadcasting logic
    standard_inputs = [seeds, Y, x, v, h]

    iterator = tqdm(range(0, N, BATCH_SIZE), desc="Batched Resample", leave=False)
    
    for i in iterator:
        start, end = i, min(i + BATCH_SIZE, N)
        actual_len = end - start
        
        # 1. Slice the standard inputs (list comprehension saves 5 lines)
        batch_args = [arr[start:end] for arr in standard_inputs]
        
        # 2. Handle s_0 logic
        if is_s0_per_seq:
            batch_s_0 = s_0[start:end]
        elif hasattr(s_0, 'shape'):
            batch_s_0 = jnp.repeat(s_0[None, ...], actual_len, axis=0)
        else:
            batch_s_0 = jnp.full((actual_len,), s_0)

        # 3. Pad if necessary
        if actual_len < BATCH_SIZE:
            batch_args = [pad_arr(a, BATCH_SIZE) for a in batch_args]
            batch_s_0  = pad_arr(batch_s_0, BATCH_SIZE)

        # 4. Run Kernel (Use * unpacking for the standard args)
        # Signature: (seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0)
        s_out = _resample_scales_kernel_batched(
            *batch_args, Cd, sigmasq, nu_s, batch_s_0
        )
        s_out.block_until_ready()

        # 5. Store (slice off padding if needed)
        results.append(s_out[:actual_len])

    return jnp.concatenate(results, axis=0)

def apply_optimizations():
    print(f"--- Applying Custom Kernels ---")
    from jax_moseq.models.keypoint_slds import gibbs
    gibbs.resample_scales = batched_resample_scales
    print(f"[Kernel] Optimized 'resample_scales' active")
    print("-------------------------------")