"""Historical in-place repetition of per-sample model output arrays."""

from typing import Dict
import numpy as np

def tile_results(results: Dict[str, dict], factor: int, verbose: bool = False) -> None:
    """
    Tiles all 1D/2D numpy arrays in the results dictionary by an integer factor in place.
    """
    if factor <= 1:
        return
    if verbose:
        print(f"[sampling] Tiling Results by factor {factor}...")
    for k in results:
        for key, val in results[k].items():
            if isinstance(val, (np.ndarray, list)):
                a = np.asarray(val)
                if a.ndim == 1:
                    results[k][key] = np.repeat(a, factor)
                else:
                    results[k][key] = np.repeat(a, factor, axis=0)



__all__ = ["tile_results"]
