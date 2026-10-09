import os
import sys
import time
import pickle
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import jax.random as jr

DEFAULT_CACHE_PATH = "/tmp/karen_slds_init_state.pkl"

def save_init_state(model, data, metadata, cache_path=DEFAULT_CACHE_PATH):
    """
    Saves the preprocessed initial (model, data, metadata) state to disk.
    This enables instant 1-second loading for all future benchmark & training runs.
    """
    print(f"[Cache] Serializing initial state to {cache_path}...")
    t0 = time.perf_counter()
    
    # Convert JAX DeviceArrays to CPU NumPy for clean, pickleable, non-CUDA serialization
    def to_cpu(obj):
        if hasattr(obj, "device"):
            return np.asarray(obj)
        elif isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_cpu(v) for v in obj]
        return obj

    clean_state = {
        "model": to_cpu(model),
        "data": to_cpu(data),
        "metadata": metadata
    }

    with open(cache_path, "wb") as f:
        pickle.dump(clean_state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    t_tot = time.perf_counter() - t0
    size_mb = os.path.getsize(cache_path) / (1024 ** 2)
    print(f"[Cache] Saved {size_mb:.2f} MB to {cache_path} in {t_tot:.2f} seconds.")


def load_init_state(cache_path=DEFAULT_CACHE_PATH, put_on_device=True):
    """
    Instantly loads the preprocessed initial state from disk.
    Bypasses 100% of data loading, outlier filtering, PCA loading, and batch initialization!
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at {cache_path}")
        
    print(f"[Cache] Loading preprocessed initial state from {cache_path}...")
    t0 = time.perf_counter()
    with open(cache_path, "rb") as f:
        state = pickle.load(f)
        
    model = state["model"]
    data = state["data"]
    metadata = state["metadata"]
    
    if put_on_device:
        # Move required tensors back to GPU device
        data = {k: jax.device_put(jnp.asarray(v)) for k, v in data.items()}
        
    t_tot = time.perf_counter() - t0
    print(f"[Cache] Successfully loaded full dataset state in {t_tot:.2f} seconds!")
    return model, data, metadata
