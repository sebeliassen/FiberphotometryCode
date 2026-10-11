"""
fpcode.kpms.init_model
======================
Batched model initialization for Keypoint-MoSeq models to avoid GPU OOM on large cohorts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def batched_init_model_by_video(
    data: dict,
    pca: dict,
    config: dict,
    params_subset_size: int = 5,
    batch_size: int = 20,
    seed: Any = None,
    location_aware: bool = False,
) -> dict:
    """
    Initializes a Keypoint-MoSeq model in batches to avoid GPU OOM errors on large datasets.
    Uses incremental seeding (jax.random.split) so each batch gets unique, uncorrelated noise.
    """
    import jax
    import keypoint_moseq as kpms

    config = dict(config, location_aware=location_aware)
    N_videos = data["Y"].shape[0]

    if seed is None:
        seed = jax.random.PRNGKey(0)
    elif isinstance(seed, int):
        seed = jax.random.PRNGKey(seed)

    # STEP 1: Initialize Global Parameters on small subset
    subset_slice = slice(0, min(params_subset_size, N_videos))
    subset_data = {
        "Y": np.asarray(data["Y"][subset_slice], dtype=np.float64),
        "mask": np.asarray(data["mask"][subset_slice], dtype=np.float64),
    }
    if "conf" in data:
        subset_data["conf"] = np.asarray(data["conf"][subset_slice], dtype=np.float64)

    def initialization_options(selection):
        options = dict(config)
        # Explicit observation-shaped priors need the same sequence slice as Y.
        prior = options.get("noise_prior")
        if hasattr(prior, "shape") and len(prior.shape) == 3 and prior.shape[0] == N_videos:
            options["noise_prior"] = prior[selection]
        return options

    model = kpms.init_model(subset_data, pca=pca, seed=seed, **initialization_options(subset_slice))

    # STEP 2: Initialize States (Batched)
    current_seed = model.get("seed", seed)
    prior = current_seed

    num_batches = int(np.ceil(N_videos / batch_size))
    full_states = {"x": [], "v": [], "h": [], "s": [], "z": []}
    full_noise_priors = []
    noise_shapes = []

    for i in range(num_batches):
        prior, batch_seed = jax.random.split(prior)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N_videos)

        batch_data = {
            "Y": np.asarray(data["Y"][start_idx:end_idx], dtype=np.float64),
            "mask": np.asarray(data["mask"][start_idx:end_idx], dtype=np.float64),
        }
        if "conf" in data:
            batch_data["conf"] = np.asarray(data["conf"][start_idx:end_idx], dtype=np.float64)

        batch_options = initialization_options(slice(start_idx, end_idx))
        # Cd includes the selected subset's whitening convention. Reinitializing
        # parameters per batch would put its states in a different latent basis.
        batch_options.update(params=model["params"], hypparams=model["hypparams"])
        batch_model = kpms.init_model(batch_data, pca=pca, seed=batch_seed, **batch_options)
        for k in full_states:
            if k in batch_model.get("states", {}):
                full_states[k].append(batch_model["states"][k])
        full_noise_priors.append(np.asarray(batch_model["noise_prior"]))
        noise_shapes.append(batch_data["Y"].shape[:-1])

    for k in full_states:
        if full_states[k]:
            full_states[k] = np.concatenate(full_states[k], axis=0)

    model["states"] = full_states
    if all(prior.ndim == 0 and prior == full_noise_priors[0] for prior in full_noise_priors):
        # Keep broadcast scalars compact when no per-observation prior is needed.
        model["noise_prior"] = full_noise_priors[0].item()
    else:
        model["noise_prior"] = np.concatenate([
            np.broadcast_to(prior, shape) for prior, shape in zip(full_noise_priors, noise_shapes)
        ], axis=0)
    return model
