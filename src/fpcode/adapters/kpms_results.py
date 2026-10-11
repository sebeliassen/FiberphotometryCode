"""Read selected KPMS results without importing the model backend."""

from contextlib import contextmanager

import h5py
import numpy as np


RESULT_FIELDS = ("syllable", "centroid", "heading", "latent_state")


class _ResultReader:
    """Read one recording at a time while the caller owns the file scope."""

    def __init__(self, source, keys):
        self.source, self.keys = source, keys

    def __iter__(self):
        return iter(self.source)

    def __contains__(self, name):
        return name in self.source

    def __getitem__(self, name):
        group = self.source[name]
        if not isinstance(group, h5py.Group) or "syllable" not in group:
            raise ValueError(f"{name!r} is not a KPMS result recording")
        result = {key: np.array(group[key]) for key in self.keys if key in group}
        if "syllable" not in result:
            raise ValueError("selected keys must include syllable")
        labels = result["syllable"]
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("syllable labels must be a one-dimensional integer array")
        if any(value.ndim < 1 or len(value) != len(labels) for value in result.values()):
            raise ValueError(f"result fields have different row counts for {name!r}")
        for value in result.values():
            value.setflags(write=False)
        return result


@contextmanager
def open_kpms_results(path, *, keys=RESULT_FIELDS):
    """Yield a reader of immutable arrays, loading one selected recording at a time."""
    with h5py.File(path, "r") as source:
        yield _ResultReader(source, tuple(keys))


def read_kpms_results(path, *, recordings=None, keys=RESULT_FIELDS):
    with open_kpms_results(path, keys=keys) as source:
        names = list(source) if recordings is None else list(recordings)
        if len(names) != len(set(names)):
            raise ValueError("recording selection contains duplicates")
        return {name: source[name] for name in names}


def results_to_frame_table(result, *, index=None):
    """Project KPMS arrays to the historical SessionRecord.syllables columns."""
    import pandas as pd

    missing = set(RESULT_FIELDS) - set(result)
    if missing:
        raise ValueError(f"KPMS result is missing required fields: {', '.join(sorted(missing))}")
    labels = np.asarray(result["syllable"])
    latent, centroid, heading = (np.asarray(result[key]) for key in ("latent_state", "centroid", "heading"))
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("syllable labels must be a one-dimensional integer array")
    if latent.ndim != 2 or centroid.ndim != 2 or centroid.shape[1] < 2 or heading.ndim != 1:
        raise ValueError("latent_state and centroid must be matrices, and heading must be a vector")
    if any(len(value) != len(labels) for value in (latent, centroid, heading)):
        raise ValueError("KPMS result fields must have matching frame counts")
    index = pd.RangeIndex(len(labels)) if index is None else index
    if len(index) != len(labels):
        raise ValueError("frame index must match the result row count")
    columns = {"syllable": pd.Series(labels, index=index, dtype="Int64")}
    columns.update({f"latent_state_{d}": latent[:, d] for d in range(latent.shape[1])})
    columns.update(centroid_0=centroid[:, 0], centroid_1=centroid[:, 1], heading=heading)
    return pd.DataFrame(columns, index=index)
