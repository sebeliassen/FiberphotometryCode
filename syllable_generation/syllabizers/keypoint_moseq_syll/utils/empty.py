import jax
import jax.numpy as jnp
import jax.random as jr
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error
from tqdm import tqdm


def _single_sequence_resample(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    """
    Logic for a SINGLE sequence.
    """
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

# Vmap over axis 0 for seed, Y, x, v, h.
# s_0 is tricky: we set in_axes=0 assuming we will format it to be per-sequence.
_kernel_vmap = jax.vmap(
    _single_sequence_resample,
    in_axes=(0, 0, 0, 0, 0, None, None, None, 0)
)

@jax.jit
def _resample_scales_kernel_batched(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    return _kernel_vmap(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0)

def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    BATCH_SIZE = 10
    N = Y.shape[0]
    seeds = jr.split(seed, N)
    results = []

    # Handle s_0 (Noise Prior) - can be scalar or per-sequence array
    is_s0_per_seq = (hasattr(s_0, "shape") and s_0.shape[0] == N)

    # We use a loop with padding to ensure JAX compiles the kernel exactly once
    iterator = tqdm(range(0, N, BATCH_SIZE), desc="Batched Resample", leave=False)

    for i in iterator:
        start = i
        end = min(i + BATCH_SIZE, N)
        actual_batch_len = end - start

        # Slicing basic inputs
        batch_seeds = seeds[start:end]
        batch_Y = Y[start:end]
        batch_x = x[start:end]
        batch_v = v[start:end]
        batch_h = h[start:end]

        # --- HANDLING s_0 ---
        # If s_0 is global (e.g. shape (K,)), we must REPEAT it for vmap to work
        # because in_axes=0 expects the 0th dim to match the batch size.
        if is_s0_per_seq:
            batch_s_0 = s_0[start:end]
        else:
            # If s_0 is scalar or shape (K,), broadcast it to (Batch, ...)
            if hasattr(s_0, 'shape'):
                # s_0 is array (K,) -> make (Batch, K,)
                batch_s_0 = jnp.repeat(s_0[None, ...], actual_batch_len, axis=0)
            else:
                # s_0 is scalar -> make (Batch,)
                batch_s_0 = jnp.full((actual_batch_len,), s_0)

        # --- PADDING ---
        # We pad inputs so they ALWAYS match BATCH_SIZE.
        # This prevents JAX from recompiling on the last uneven batch.
        if actual_batch_len < BATCH_SIZE:
            pad_len = BATCH_SIZE - actual_batch_len

            def pad_arr(arr):
                # pad 0th dim with edge values
                padding = [(0, pad_len)] + [(0, 0)] * (arr.ndim - 1)
                return jnp.pad(arr, padding, mode='edge')

            batch_seeds_in = pad_arr(batch_seeds)
            batch_Y_in = pad_arr(batch_Y)
            batch_x_in = pad_arr(batch_x)
            batch_v_in = pad_arr(batch_v)
            batch_h_in = pad_arr(batch_h)
            batch_s_0_in = pad_arr(batch_s_0)
        else:
            batch_seeds_in = batch_seeds
            batch_Y_in = batch_Y
            batch_x_in = batch_x
            batch_v_in = batch_v
            batch_h_in = batch_h
            batch_s_0_in = batch_s_0

        # Run Kernel
        s_out_padded = _resample_scales_kernel_batched(
            batch_seeds_in,
            batch_Y_in,
            batch_x_in,
            batch_v_in,
            batch_h_in,
            Cd,
            sigmasq,
            nu_s,
            batch_s_0_in
        )

        # Force sync to catch OOMs immediately per batch
        s_out_padded.block_until_ready()

        # Un-pad and store
        if actual_batch_len < BATCH_SIZE:
            results.append(s_out_padded[:actual_batch_len])
        else:
            results.append(s_out_padded)

    return jnp.concatenate(results, axis=0)
