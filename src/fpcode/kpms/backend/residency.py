"""KPMS residency; numerical algorithms preserved from the baseline."""

from __future__ import annotations

from jax.tree_util import tree_flatten_with_path
import jax
import jax.numpy as jnp
import numpy as np


def fast_consolidated_check_for_nans(data):
    """
    Consolidated NaN check across all CPU and GPU arrays in a pytree.
    Evaluates GPU arrays on-device, combining them into a single GPU boolean flag
    fetched to host once per iteration. CPU arrays are checked directly on host.
    Detailed error messages and path formatting are only computed on failure.
    """
    def _format(path, num_nan):
        path = "/".join(map(str, path))
        return f"{num_nan} NaNs found in {path}"

    flat_res = tree_flatten_with_path(data)
    if len(flat_res) == 3:
        paths_and_leaves = list(zip(flat_res[0], flat_res[1]))
    else:
        paths_and_leaves = flat_res[0]
    gpu_leaves = []
    cpu_leaves = []

    for path, value in paths_and_leaves:
        if hasattr(value, "device"):  # JAX array
            dev_str = str(value.device).lower()
            if "gpu" in dev_str or "cuda" in dev_str:
                gpu_leaves.append(value)
            else:
                cpu_leaves.append(value)
        elif isinstance(value, np.ndarray):
            cpu_leaves.append(value)

    # 1. Check GPU arrays in single reduction on-device
    has_gpu_nan = False
    if gpu_leaves:
        gpu_any_flags = [jnp.isnan(arr).any() for arr in gpu_leaves]
        combined_gpu_flag = jnp.any(jnp.stack(gpu_any_flags))
        has_gpu_nan = bool(combined_gpu_flag)

    # 2. Check CPU arrays on host CPU
    has_cpu_nan = False
    if cpu_leaves:
        for arr in cpu_leaves:
            arr_np = np.asarray(arr)
            if np.isnan(arr_np).any():
                has_cpu_nan = True
                break

    # Fast path: 0 NaNs anywhere
    if not has_gpu_nan and not has_cpu_nan:
        return False, [], []

    # Slow path (error only): pinpoint exact arrays and count NaNs
    nan_info = []
    messages = []
    for path, value in paths_and_leaves:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            val_np = np.asarray(value)
            if np.isnan(val_np).any():
                num_nans = int(np.isnan(val_np).sum())
                nan_info.append((path, num_nans))
                messages.append(_format(path, num_nans))

    return True, nan_info, messages


def safe_device_put_as_scalar(tree):
    """
    Selective device_put_as_scalar that preserves host CPU residency for massive
    arrays like states['s'] (3.42 GB) and noise_prior (3.42 GB).
    """
    def put_leaf(path, leaf):
        path_keys = [getattr(p, 'key', str(p)) for p in path]
        if ('states' in path_keys and 's' in path_keys) or ('noise_prior' in path_keys):
            if hasattr(leaf, 'shape') and leaf.shape == ():
                return leaf.item()
            if isinstance(leaf, np.ndarray):
                return leaf
            return jax.device_put(leaf, jax.devices('cpu')[0])
        else:
            dev_leaf = jax.device_put(leaf)
            if hasattr(dev_leaf, 'shape') and dev_leaf.shape == ():
                return dev_leaf.item()
            return dev_leaf

    return jax.tree_util.tree_map_with_path(put_leaf, tree)
