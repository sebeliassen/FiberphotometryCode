"""
fpcode.behavior.transitions
===========================
Bigram Markov transition matrices, transition graph analysis,
and stationary distributions for syllable sequences.
"""

from typing import List, Optional, Union

import numpy as np


def compute_transition_matrix(
    syllables: Union[np.ndarray, List[int], List[np.ndarray]],
    num_states: Optional[int] = None,
    normalize: bool = True,
    add_pseudocount: float = 0.0,
) -> np.ndarray:
    """
    Computes empirical state-to-state transition probability matrix P[i, j] = P(s_{t+1} = j | s_t = i).
    Accepts single sequence or list of sequences across recordings.
    """
    if isinstance(syllables, list) and len(syllables) > 0 and isinstance(syllables[0], (list, np.ndarray)):
        sequences = [np.asarray(s, dtype=int) for s in syllables]
    else:
        sequences = [np.asarray(syllables, dtype=int)]

    if num_states is None:
        max_state = max((np.max(s) for s in sequences if len(s) > 0), default=-1)
        num_states = max_state + 1

    counts = np.full((num_states, num_states), add_pseudocount, dtype=np.float64)

    for seq in sequences:
        if len(seq) < 2:
            continue
        valid = (seq[:-1] >= 0) & (seq[:-1] < num_states) & (seq[1:] >= 0) & (seq[1:] < num_states)
        u = seq[:-1][valid]
        v = seq[1:][valid]
        np.add.at(counts, (u, v), 1.0)

    if not normalize:
        return counts

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    return P


def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Computes stationary distribution pi such that pi @ P = pi and sum(pi) = 1.
    Uses leading left eigenvector of the transition matrix.
    """
    P = np.asarray(transition_matrix, dtype=np.float64)
    n = P.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvalue closest to 1.0
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, idx])
    s_sum = stationary.sum()
    if s_sum != 0:
        stationary = stationary / s_sum
    return np.maximum(stationary, 0.0)
