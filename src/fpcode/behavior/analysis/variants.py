"""Group-blind robust clustering of observed bout execution measurements."""

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment


def fit_execution_variants(values, *, clusters=2, seed=101):
    """Fit standardized numerical variants; clusters do not carry clinical labels.

    Callers select training animals and balanced observations. Duration and
    movement amplitude remain independent input features. Returns raw-unit
    centers, scaling, repeat-seed assignment agreement and explained dispersion.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < clusters * 10 or not np.isfinite(values).all():
        raise ValueError("insufficient finite training measurements")
    location = np.median(values, axis=0)
    scale = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scale = np.where(scale > 1e-9, scale, np.std(values, axis=0))
    scale = np.maximum(scale, 1e-9)
    x = np.clip((values - location) / scale, -10, 10)
    center, labels = kmeans2(x, clusters, iter=50, minit="++", seed=seed)
    other_center, other = kmeans2(x, clusters, iter=50, minit="++", seed=seed + 1)
    contingency = np.zeros((clusters, clusters))
    np.add.at(contingency, (labels, other), 1)
    a, b = linear_sum_assignment(-contingency)
    total = np.sum((x - x.mean(axis=0)) ** 2)
    return {"centers": center, "location": location, "scale": scale,
            "raw_centers": center * scale + location,
            "seed_assignment_agreement": float(contingency[a, b].sum() / len(values)),
            "explained_dispersion": float(1 - np.sum((x - center[labels]) ** 2) / total) if total else 0.}


def assign_execution_variants(values, model):
    x = np.clip((np.asarray(values) - model["location"]) / model["scale"], -10, 10)
    return np.argmin(np.sum((x[:, None, :] - model["centers"][None, :, :]) ** 2, axis=2), axis=1)
