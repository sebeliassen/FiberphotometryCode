"""Collect precomputed trial matrices without owning session construction."""

from collections import defaultdict

import numpy as np

from fpcode.adapters.legacy_sessions import iter_signal_matrices


def group_signals(sessions, key_fn, *, missing="skip"):
    groups = defaultdict(list)
    for session in sessions:
        for signal, matrix in iter_signal_matrices(session, missing=missing):
            groups[key_fn(session, signal)].append(matrix)
    return {key: np.vstack(matrices) for key, matrices in groups.items() if matrices}
