"""Read selected KPMS results using HDF5 without importing the model backend."""

import h5py
import numpy as np


def read_kpms_results(path, *, recordings=None, keys=("syllable", "centroid", "heading", "latent_state")):
    out = {}
    with h5py.File(path, "r") as source:
        names = list(source) if recordings is None else list(recordings)
        if len(names) != len(set(names)):
            raise ValueError("recording selection contains duplicates")
        for name in names:
            group = source[name]
            if not isinstance(group, h5py.Group) or "syllable" not in group:
                raise ValueError(f"{name!r} is not a KPMS result recording")
            result = {key: np.array(group[key]) for key in keys if key in group}
            if "syllable" not in result:
                raise ValueError("selected keys must include syllable")
            labels = result["syllable"]
            if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("syllable labels must be a one-dimensional integer array")
            if any(value.ndim < 1 or len(value) != len(labels) for value in result.values()):
                raise ValueError(f"result fields have different row counts for {name!r}")
            for value in result.values():
                value.setflags(write=False)
            out[name] = result
    return out
