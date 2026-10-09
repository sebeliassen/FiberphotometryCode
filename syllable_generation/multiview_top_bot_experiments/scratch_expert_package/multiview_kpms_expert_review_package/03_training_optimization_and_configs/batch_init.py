import jax
import jax.numpy as jnp
import numpy as np
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds.initialize import init_states, estimate_error
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca

def batched_init_model_by_video(data, pca, config, params_subset_size=5, batch_size=None, seed=None):
    """
    Initializes a Keypoint-MoSeq model in batches to avoid GPU OOM errors on large datasets.
    
    This version uses incremental seeding (jax.random.split) to ensure each batch 
    gets unique, uncorrelated random noise during state initialization.
    """
    if batch_size is None:
        try:
            from utils.resample_batching import CONFIG
            batch_size = min(CONFIG.discrete_bs, 20)
        except Exception:
            batch_size = 20

    N_videos = data['Y'].shape[0]

    if seed is None:
        seed = jax.random.PRNGKey(0)

    # STEP 1: Initialize Global Parameters (Small subset)
    # We use a small slice of data to fit the global params (PCA, noise prior, etc.)
    subset_slice = slice(0, min(params_subset_size, N_videos))
    subset_data = {
        'Y': np.asarray(data['Y'][subset_slice], dtype=np.float64),
        'mask': np.asarray(data['mask'][subset_slice], dtype=np.float64),
        'conf': np.asarray(data['conf'][subset_slice], dtype=np.float64) if 'conf' in data else None
    }

    # Initialize the "skeleton" model with global params
    model = kpms.init_model(subset_data, pca=pca, seed=seed, **config)

    # STEP 2: Initialize States (Batched)
    params = model['params']
    hypparams = model['hypparams']
    obs_hypparams = hypparams['obs_hypparams']
    
    # Use model's seed if it returned one; otherwise keep the input seed
    current_seed = model.get('seed', seed)
    
    # Setup for incremental seeding
    prior = current_seed

    num_batches = int(np.ceil(N_videos / batch_size))
    full_states = {'x': [], 'v': [], 'h': [], 's': [], 'z': []}
    
    print(f"[Batch Init] Initializing states for {N_videos} videos in {num_batches} batches...")

    for i in range(num_batches):
        # Generate a unique key for this batch to prevent correlated initialization
        prior, batch_seed = jax.random.split(prior)

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N_videos)

        Y_chunk = jnp.array(data['Y'][start_idx:end_idx])
        mask_chunk = jnp.array(data['mask'][start_idx:end_idx])
        conf_chunk = jnp.array(data['conf'][start_idx:end_idx]) if 'conf' in data else None

        # Manual Preprocessing for this chunk
        Y_flat_chunk, v_chunk, h_chunk = preprocess_for_pca(
            Y_chunk,
            anterior_idxs=config.get('anterior_idxs'),
            posterior_idxs=config.get('posterior_idxs'),
            conf=conf_chunk,
            conf_threshold=config.get('conf_threshold', 0.5),
            fix_heading=config.get('fix_heading', False),
            verbose=False
        )

        # Handle noise prior for this chunk
        if conf_chunk is not None and config.get('error_estimator', None) is not None:
            batch_noise_prior = estimate_error(conf_chunk, **config['error_estimator'])
        else:
            # Fallback if no error estimator config is present
            batch_noise_prior = 1.0

        # Initialize states for the chunk
        chunk_states = init_states(
            batch_seed,
            Y_chunk,
            mask_chunk,
            params,
            batch_noise_prior,
            obs_hypparams,
            Y_flat=Y_flat_chunk,
            v=v_chunk,
            h=h_chunk,
        )

        # Offload results to CPU immediately to free VRAM
        for k in ['x', 'v', 'h', 's', 'z']:
            if k in chunk_states:
                full_states[k].append(jax.device_put(chunk_states[k], jax.devices('cpu')[0]))

        # Explicit cleanup to help JAX memory management
        del Y_chunk, mask_chunk, conf_chunk, Y_flat_chunk, v_chunk, h_chunk, chunk_states, batch_noise_prior

        if (i+1) % 5 == 0 or (i+1) == num_batches:
            print(f"  Batch {i+1}/{num_batches} done.")

    # STEP 3: Stitching
    print("[Batch Init] Concatenating all states...")
    final_states = {}
    gpu_device = jax.devices()[0]
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        for k in ['x', 'v', 'h', 's', 'z']:
            if full_states[k]:
                concat_k = jnp.concatenate(full_states[k], axis=0)
                final_states[k] = jax.device_put(concat_k, gpu_device)

    model['states'] = final_states

    # Recalculate full noise_prior if needed (on CPU)
    # Keeping noise_prior on CPU saves 3.4 GB VRAM on full cohort; batched_resample_scales streams slices on demand
    if 'conf' in data and config.get('error_estimator', None) is not None:
        print("[Batch Init] Recalculating full noise_prior (on CPU)...")
        with jax.default_device(cpu_device):
            model['noise_prior'] = estimate_error(jnp.array(data['conf']), **config['error_estimator'])
    else:
        model['noise_prior'] = 1.0

    # Store the final PRNGKey state for reproducibility
    model['seed'] = prior

    print(f"[Batch Init] Done. State 'x' shape: {final_states['x'].shape}")
    return model