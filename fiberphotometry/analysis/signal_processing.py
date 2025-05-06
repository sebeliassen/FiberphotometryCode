# analysis/signal_processing.py

import numpy as np
from collections import defaultdict
from scipy import stats

def group_signals(sessions, key_fn):
    """
    Collect all non-empty signal_matrix arrays into groups.
    key_fn: (session, signal_key) → grouping key (e.g. (genotype, event, region))
    Returns dict: key → np.vstack(list_of_matrices)
    """
    groups = defaultdict(list)
    for s in sessions:
        for sig_key, info in s.signal_info.items():
            mat = info.get('signal_matrix')
            if mat is not None and mat.size:
                groups[key_fn(s, sig_key)].append(mat)
    return {k: np.vstack(v) for k, v in groups.items() if v}

def compute_mean_ci(mat, alpha=0.95):
    """
    Given mat shape (n_trials, n_time), return (mean, lower_ci, upper_ci).
    """
    ys = np.mean(mat, axis=0)
    sem = np.std(mat, axis=0) / np.sqrt(mat.shape[0])
    lo, hi = stats.norm.interval(alpha, loc=ys, scale=sem)
    return ys, lo, hi
