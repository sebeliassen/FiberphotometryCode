"""Generic sequence inference: stable log-domain HMM and explicit-duration HSMM.

Provides forward-backward and Viterbi algorithms for categorical/array sequence
models using float64 numerics, exact structural zeros (-inf), and stable log-sum-exp.
Includes segment-based HSMM recursion with declared duration PMFs and boundary survival.
"""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray | float:
    """Compute log(sum(exp(a))) stably, handling -inf correctly."""
    return _scipy_logsumexp(np.asarray(a, dtype=np.float64), axis=axis, keepdims=keepdims)


def validate_log_probabilities(values, *, axis=-1, name="probabilities", allow_terminal=False):
    """Validate declared probability tables without flooring structural zeros."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.any(np.isnan(values) | np.isposinf(values)) or np.any(values > 0):
        raise ValueError(f"{name} must contain finite log probabilities or -inf")
    normalizer = logsumexp(values, axis=axis)
    valid = np.isclose(normalizer, 0., atol=1e-6, rtol=0)
    if allow_terminal:
        valid |= np.isneginf(normalizer)
    if not np.all(valid):
        raise ValueError(f"{name} must be normalized on its declared support")


def _validate_inputs(initial, transition, emission, duration=None, *, memory_budget_bytes=256*1024*1024,
                     max_work=100000000):
    if emission.ndim != 2 or emission.shape[1] < 1:
        raise ValueError("emissions must have shape (observations, positive state count)")
    t, q = emission.shape
    if initial.shape != (q,) or transition.shape != (q, q):
        raise ValueError("initial/transition shapes disagree with emission states")
    if any(np.any(np.isnan(a) | np.isposinf(a)) for a in (initial, transition, emission)):
        raise ValueError("inference arrays must contain finite values or structural -inf")
    validate_log_probabilities(initial, axis=0, name="initial distribution")
    validate_log_probabilities(transition, name="transition distribution", allow_terminal=duration is not None)
    d = 1
    if duration is not None:
        if duration.ndim != 2 or duration.shape[0] != q or duration.shape[1] < 1:
            raise ValueError("duration PMFs must have shape (states, positive duration support)")
        validate_log_probabilities(duration, name="duration PMFs")
        if np.any(~np.isneginf(np.diag(transition))):
            raise ValueError("HSMM segment self transitions must be structural -inf")
        d = duration.shape[1]
    for value, name in ((memory_budget_bytes, "memory budget"), (max_work, "work budget")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Viterbi calls forward while its traceback arrays are still live. Explicit
    # durations also allocate survival storage and duration-by-state batches.
    temporal_bytes = (96 if duration is not None else 64)*(t+1)*q
    duration_bytes = 0 if duration is None else 16*q*d + 64*min(d, t)*q + 32*min(d, t)
    if temporal_bytes + 16*q*q + 256*t + duration_bytes > memory_budget_bytes:
        raise ValueError("sequence inference memory budget exceeded")
    if max(t, 1)*q*q*d > max_work:
        raise ValueError("sequence inference work budget exceeded")


def _segment_emissions(emission):
    forbidden = np.isneginf(emission)
    sums = np.vstack((np.zeros(emission.shape[1]), np.cumsum(np.where(forbidden, 0., emission), axis=0)))
    counts = np.vstack((np.zeros(emission.shape[1], dtype=np.int64), np.cumsum(forbidden, axis=0)))
    return sums, counts


def _segment_score(prefix, counts, start, stop, state):
    return -np.inf if counts[stop, state] != counts[start, state] else float(prefix[stop, state]-prefix[start, state])


class HMMResult(NamedTuple):
    log_likelihood: float
    log_forward: np.ndarray
    log_backward: np.ndarray
    log_posteriors: np.ndarray
    posteriors: np.ndarray


def forward_backward_hmm(
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_emission: np.ndarray,
    *,
    memory_budget_bytes: int = 256*1024*1024,
    max_work: int = 100000000,
) -> HMMResult:
    """Forward-backward algorithm for standard discrete-time HMM.

    Parameters
    ----------
    log_initial : np.ndarray
        Shape (Q,), initial state log probabilities.
    log_transition : np.ndarray
        Shape (Q, Q), log transition matrix A[i, j] = log P(s_t = j | s_{t-1} = i).
    log_emission : np.ndarray
        Shape (T, Q), log emission probabilities B[t, j] = log P(o_t | s_t = j).

    Returns
    -------
    HMMResult
        Likelihood, forward/backward log matrices, and normalized state posteriors.
    """
    log_init = np.asarray(log_initial, dtype=np.float64)
    log_trans = np.asarray(log_transition, dtype=np.float64)
    log_emiss = np.asarray(log_emission, dtype=np.float64)

    _validate_inputs(log_init, log_trans, log_emiss, memory_budget_bytes=memory_budget_bytes, max_work=max_work)
    T, Q = log_emiss.shape
    if log_init.shape != (Q,) or log_trans.shape != (Q, Q):
        raise ValueError(f"shape mismatch: init {log_init.shape}, trans {log_trans.shape}, emiss {log_emiss.shape}")
    if T == 0:
        return HMMResult(0.0, np.empty((0, Q)), np.empty((0, Q)), np.empty((0, Q)), np.empty((0, Q)))

    # Forward pass
    log_alpha = np.full((T, Q), float("-inf"), dtype=np.float64)
    log_alpha[0] = log_init + log_emiss[0]

    for t in range(1, T):
        # For each destination state j: logsumexp over i (alpha[t-1, i] + trans[i, j]) + emiss[t, j]
        # alpha[t-1, :, None] + trans: shape (Q, Q) where axis 0 is i, axis 1 is j
        step = logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0)
        log_alpha[t] = step + log_emiss[t]

    log_lik = float(logsumexp(log_alpha[-1]))

    # If impossible sequence
    if np.isneginf(log_lik):
        return HMMResult(
            float("-inf"),
            log_alpha,
            np.full((T, Q), float("-inf"), dtype=np.float64),
            np.full((T, Q), float("-inf"), dtype=np.float64),
            np.zeros((T, Q), dtype=np.float64),
        )

    # Backward pass
    log_beta = np.full((T, Q), float("-inf"), dtype=np.float64)
    log_beta[-1] = 0.0  # log(1.0) = 0

    for t in range(T - 2, -1, -1):
        # beta[t, i] = logsumexp over j (trans[i, j] + emiss[t+1, j] + beta[t+1, j])
        terms = log_trans + log_emiss[t + 1, None, :] + log_beta[t + 1, None, :]
        log_beta[t] = logsumexp(terms, axis=1)

    # Posteriors gamma[t, i] = alpha[t, i] + beta[t, i] - log_lik
    log_gamma = log_alpha + log_beta - log_lik
    posteriors = np.exp(np.clip(log_gamma, float("-inf"), 0.0))
    # Normalize row sums to 1.0 on supported rows
    row_sums = posteriors.sum(axis=1, keepdims=True)
    posteriors = np.where(row_sums > 0, posteriors / row_sums, 0.0)

    return HMMResult(
        log_lik,
        log_alpha,
        log_beta,
        log_gamma,
        posteriors,
    )


def viterbi_hmm(
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_emission: np.ndarray,
    *,
    memory_budget_bytes: int = 256*1024*1024,
    max_work: int = 100000000,
) -> tuple[np.ndarray, float]:
    """Viterbi algorithm finding MAP state sequence for discrete-time HMM.

    Returns
    -------
    tuple
        (best_path: np.ndarray of shape (T,), best_log_score: float)
    """
    log_init = np.asarray(log_initial, dtype=np.float64)
    log_trans = np.asarray(log_transition, dtype=np.float64)
    log_emiss = np.asarray(log_emission, dtype=np.float64)

    _validate_inputs(log_init, log_trans, log_emiss, memory_budget_bytes=memory_budget_bytes, max_work=max_work)
    T, Q = log_emiss.shape
    if T == 0:
        return np.empty(0, dtype=int), 0.0

    viterbi = np.full((T, Q), float("-inf"), dtype=np.float64)
    backpointer = np.zeros((T, Q), dtype=int)

    viterbi[0] = log_init + log_emiss[0]

    for t in range(1, T):
        # shape (Q, Q): rows i, cols j
        scores = viterbi[t - 1, :, None] + log_trans
        backpointer[t] = np.argmax(scores, axis=0)
        viterbi[t] = np.max(scores, axis=0) + log_emiss[t]

    best_score = float(np.max(viterbi[-1]))
    if np.isneginf(best_score):
        return np.full(T, -1, dtype=int), float("-inf")

    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(viterbi[-1])
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path, best_score


class HSMMSegment(NamedTuple):
    state: int
    start: int
    stop: int
    duration: int
    score: float


class HSMMResult(NamedTuple):
    log_likelihood: float
    segments: tuple[HSMMSegment, ...]
    map_states: np.ndarray
    best_score: float


def compute_survival_log_pmfs(log_duration_pmfs: np.ndarray) -> np.ndarray:
    """Compute tail survival log probabilities log P(D >= d) from duration log PMFs.

    log_duration_pmfs : shape (Q, D_max) where d = 1..D_max.
    Returns : shape (Q, D_max) where [i, d-1] = log P(D >= d | S=i).
    """
    log_pmf = np.asarray(log_duration_pmfs, dtype=np.float64)
    Q, D_max = log_pmf.shape
    validate_log_probabilities(log_pmf, name="duration PMFs")
    return np.logaddexp.accumulate(log_pmf[:, ::-1], axis=1)[:, ::-1]


def forward_hsmm(
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_emission: np.ndarray,
    log_duration_pmfs: np.ndarray,
    right_censored: bool = True,
    *,
    memory_budget_bytes: int = 256*1024*1024,
    max_work: int = 100000000,
) -> tuple[float, np.ndarray]:
    """Compute exact HSMM forward marginal likelihood via log-sum-exp recursion.

    Parameters
    ----------
    log_initial : np.ndarray
        Shape (Q,), log initial state probabilities.
    log_transition : np.ndarray
        Shape (Q, Q), log transition probabilities (self-transitions must be -inf).
    log_emission : np.ndarray
        Shape (T, Q), log observation probabilities at each time step.
    log_duration_pmfs : np.ndarray
        Shape (Q, D_max), log duration PMFs log P(D = d | state). d is 1-indexed (col 0 is d=1).
    right_censored : bool
        If True, final segment ending at T uses survival probability P(D >= d).

    Returns
    -------
    log_likelihood : float
        Total marginal log likelihood of the sequence summed across all legal segmentations.
    alpha : np.ndarray
        Shape (T + 1, Q), forward log probabilities ending at each time boundary.
    """
    log_init, log_trans, log_emiss, log_dur = (np.asarray(a, dtype=np.float64) for a in
        (log_initial, log_transition, log_emission, log_duration_pmfs))
    _validate_inputs(log_init, log_trans, log_emiss, log_dur,
                     memory_budget_bytes=memory_budget_bytes, max_work=max_work)
    T, Q = log_emiss.shape
    if not T:
        return 0.0, np.zeros((1, Q), dtype=np.float64)
    survival = compute_survival_log_pmfs(log_dur)
    prefix, forbidden = _segment_emissions(log_emiss)
    alpha = np.full((T+1, Q), -np.inf)
    incoming = np.full_like(alpha, -np.inf)
    incoming[0] = log_init
    for stop in range(1, T+1):
        durations = np.arange(1, min(log_dur.shape[1], stop)+1)
        starts = stop-durations
        emissions = prefix[stop]-prefix[starts]
        emissions[forbidden[stop] != forbidden[starts]] = -np.inf
        pmf = survival if right_censored and stop == T else log_dur
        alpha[stop] = logsumexp(incoming[starts]+pmf[:, durations-1].T+emissions, axis=0)
        incoming[stop] = logsumexp(alpha[stop, :, None]+log_trans, axis=0)
    return float(logsumexp(alpha[T])), alpha


def viterbi_hsmm(
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_emission: np.ndarray,
    log_duration_pmfs: np.ndarray,
    *,
    right_censored: bool = True,
    memory_budget_bytes: int = 256*1024*1024,
    max_work: int = 100000000,
) -> HSMMResult:
    """Viterbi algorithm for explicit-duration Hidden Semi-Markov Model.

    Parameters
    ----------
    log_initial : np.ndarray
        Shape (Q,), log initial state probabilities.
    log_transition : np.ndarray
        Shape (Q, Q), log segment transition matrix. Self-transitions A[i, i] must be -inf.
    log_emission : np.ndarray
        Shape (T, Q), per-step log emission probabilities.
    log_duration_pmfs : np.ndarray
        Shape (Q, D_max), log duration PMFs log P(D = d | state). d is 1-indexed (col 0 is d=1).
    right_censored : bool
        If True, final segment ending at T uses survival probability P(D >= d) rather than exact PMF P(D=d).

    Returns
    -------
    HSMMResult
        Likelihood, parsed segments, frame-by-frame MAP state assignments, and best path score.
    """
    log_init, log_trans, log_emiss, log_dur = (np.asarray(a, dtype=np.float64) for a in
        (log_initial, log_transition, log_emission, log_duration_pmfs))
    _validate_inputs(log_init, log_trans, log_emiss, log_dur,
                     memory_budget_bytes=memory_budget_bytes, max_work=max_work)
    T, Q = log_emiss.shape
    if not T:
        return HSMMResult(0.0, (), np.empty(0, dtype=int), 0.0)
    survival = compute_survival_log_pmfs(log_dur)
    prefix, forbidden = _segment_emissions(log_emiss)
    V = np.full((T+1, Q), -np.inf)
    incoming = np.full_like(V, -np.inf)
    incoming[0] = log_init
    predecessors = np.full((T+1, Q), -1, dtype=int)
    bp_state, bp_duration = np.full_like(predecessors, -1), np.zeros_like(predecessors)
    states = np.arange(Q)
    for stop in range(1, T+1):
        # Preserve historical ties: an initial segment wins before the ascending
        # subsequent durations, and the smallest predecessor state wins ties.
        durations = (np.r_[stop, np.arange(1, stop)] if stop <= log_dur.shape[1]
                     else np.arange(1, log_dur.shape[1]+1))
        starts = stop-durations
        emissions = prefix[stop]-prefix[starts]
        emissions[forbidden[stop] != forbidden[starts]] = -np.inf
        pmf = survival if right_censored and stop == T else log_dur
        candidates = incoming[starts]+pmf[:, durations-1].T+emissions
        best = np.argmax(candidates, axis=0)
        V[stop] = candidates[best, states]
        bp_duration[stop] = durations[best]
        bp_state[stop] = predecessors[starts[best], states]
        transitions = V[stop, :, None]+log_trans
        predecessors[stop] = np.argmax(transitions, axis=0)
        incoming[stop] = transitions[predecessors[stop], states]
    best_score = float(np.max(V[T]))
    if np.isneginf(best_score):
        return HSMMResult(-np.inf, (), np.full(T, -1, dtype=int), -np.inf)
    segments, path = [], np.full(T, -1, dtype=int)
    stop, state = T, int(np.argmax(V[T]))
    while stop > 0:
        duration, previous = int(bp_duration[stop, state]), int(bp_state[stop, state])
        start = stop-duration
        score = float(V[stop, state]-(V[start, previous] if previous >= 0 else 0.))
        segments.append(HSMMSegment(state, start, stop, duration, score))
        path[start:stop] = state
        stop, state = start, previous
    likelihood, _ = forward_hsmm(log_init, log_trans, log_emiss, log_dur,
        right_censored=right_censored, memory_budget_bytes=memory_budget_bytes, max_work=max_work)
    return HSMMResult(likelihood, tuple(reversed(segments)), path, best_score)


__all__ = [
    "logsumexp",
    "validate_log_probabilities",
    "HMMResult",
    "forward_backward_hmm",
    "viterbi_hmm",
    "HSMMSegment",
    "HSMMResult",
    "compute_survival_log_pmfs",
    "forward_hsmm",
    "viterbi_hsmm",
]
