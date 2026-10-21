"""Probability divergence with the explicitly preserved presentation policy."""

import numpy as np


def js_divergence(P, Q, eps=1e-12):
    """Return Jensen-Shannon divergence in bits along the last axis.

    Preserve the historical policy: clip each weight at eps before independently
    normalizing P and Q, then use base-2 logarithms. Inputs are not modified.
    This is divergence, rather than the square-root distance.
    """
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)
    P = P / P.sum(axis=-1, keepdims=True)
    Q = Q / Q.sum(axis=-1, keepdims=True)
    M = 0.5 * (P + Q)
    return 0.5 * np.sum(P * np.log2(P / M), axis=-1) + 0.5 * np.sum(Q * np.log2(Q / M), axis=-1)
