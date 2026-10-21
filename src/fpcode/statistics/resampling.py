"""Seeded bootstrap summaries over independent observations."""

import numpy as np


def bootstrap_mean_interval(values, *, seed, n_resamples=1000, confidence=0.95):
    """Percentile interval; callers define the independent experimental unit.

    No session/frame resampling policy is inferred. Nonfinite observations are
    rejected so exclusion decisions remain explicit at the study layer.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap observations must be a nonempty finite vector")
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    # Bound temporary allocations even if a caller requests many resamples.
    means = np.empty(int(n_resamples))
    for start in range(0, len(means), 128):
        stop = min(start + 128, len(means))
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(low), float(high)
