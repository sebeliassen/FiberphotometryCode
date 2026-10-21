"""CPU convolutional NMF with data-dependent seqNMF cross-factor competition.

Independent implementation of the multiplicative updates described by
Mackevicius et al. (eLife 2019, doi:10.7554/eLife.38471). Uses zero shifts and
explicit valid onset masks, not circular joins. No backend initializes on import.
This is not a claim of numerical identity with the MATLAB reference.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


def reconstruct(w, h):
    result = np.zeros((w.shape[0], h.shape[1]), dtype=np.float32)
    for lag in range(w.shape[2]):
        if lag < h.shape[1]:
            result[:, lag:] += w[:, :, lag] @ h[:, :h.shape[1] - lag]
    return result


def _adjoint(w, x):
    result = np.zeros((w.shape[1], x.shape[1]), dtype=np.float32)
    for lag in range(w.shape[2]):
        if lag < x.shape[1]:
            result[:, :x.shape[1] - lag] += w[:, :, lag].T @ x[:, lag:]
    return result


def _smooth(x, length):
    width = 2 * length - 1
    return uniform_filter1d(x, width, axis=1, mode="constant") * width


class SeqNMF:
    """Explicitly invoked, seeded factorization for nonnegative feature timelines."""

    def __init__(self, factors, length, *, penalty, iterations=80, seed=101):
        if factors < 1 or length < 2 or iterations < 1 or penalty < 0:
            raise ValueError("invalid seqNMF settings")
        self.factors, self.length = factors, length
        self.penalty, self.iterations, self.seed = penalty, iterations, seed

    def fit(self, x, *, valid_starts=None):
        x = self._validate(x)
        valid = self._valid_starts(x.shape[1], valid_starts)
        rng = np.random.default_rng(self.seed)
        w = rng.uniform(.01, .1, (x.shape[0], self.factors, self.length)).astype(np.float32)
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            raise ValueError("no supported motif windows")
        for k, start in enumerate(rng.choice(candidates, self.factors, replace=True)):
            w[:, k, :] += x[:, start:start + self.length]
        h = rng.uniform(.001, .01, (self.factors, x.shape[1])).astype(np.float32)
        h[:, ~valid] = 0
        smoothed_x = _smooth(x, self.length)
        self.loss_history = []
        for iteration in range(self.iterations):
            prediction = reconstruct(w, h)
            numerator = _adjoint(w, x)
            smooth_match = _smooth(numerator, self.length)
            competition = smooth_match.sum(axis=0, keepdims=True) - smooth_match
            h *= numerator / np.maximum(_adjoint(w, prediction) + self.penalty * competition, 1e-9)
            h[:, ~valid] = 0
            prediction = reconstruct(w, h)
            for lag in range(self.length):
                hs = h[:, :x.shape[1] - lag]
                nx = x[:, lag:] @ hs.T
                denominator = prediction[:, lag:] @ hs.T
                if self.penalty:
                    cross = smoothed_x[:, lag:] @ hs.T
                    denominator += self.penalty * (cross.sum(axis=1, keepdims=True) - cross)
                w[:, :, lag] *= nx / np.maximum(denominator, 1e-9)
            norms = np.sqrt(np.sum(w ** 2, axis=(0, 2)))
            w /= np.maximum(norms[None, :, None], 1e-9)
            h *= norms[:, None]
            if iteration % 10 == 0 or iteration == self.iterations - 1:
                self.loss_history.append(float(np.linalg.norm(x - reconstruct(w, h)) / np.linalg.norm(x)))
        self.w, self.h = w, h
        return self

    def infer(self, x, *, valid_starts=None, iterations=50):
        x = self._validate(x)
        if x.shape[0] != self.w.shape[0]:
            raise ValueError("inference vocabulary differs from template vocabulary")
        valid = self._valid_starts(x.shape[1], valid_starts)
        h = np.full((self.factors, x.shape[1]), .01, dtype=np.float32)
        h[:, ~valid] = 0
        numerator = _adjoint(self.w, x)
        smooth_match = _smooth(numerator, self.length)
        competition = smooth_match.sum(axis=0, keepdims=True) - smooth_match
        for _ in range(iterations):
            h *= numerator / np.maximum(_adjoint(self.w, reconstruct(self.w, h))
                                        + self.penalty * competition, 1e-9)
            h[:, ~valid] = 0
        return h

    def _valid_starts(self, n, supplied):
        valid = np.ones(n, dtype=bool) if supplied is None else np.asarray(supplied, dtype=bool).copy()
        if valid.shape != (n,):
            raise ValueError("onset mask must match the timeline")
        valid[max(0, n - self.length + 1):] = False
        return valid

    @staticmethod
    def _validate(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or not np.isfinite(x).all() or np.any(x < 0) or not np.any(x > 0):
            raise ValueError("features must be a nonempty finite nonnegative matrix")
        return x


def template_similarity(first, second):
    """Hungarian-matched cosine similarity, explicitly phase-locked."""
    a, b = first.transpose(1, 0, 2).reshape(first.shape[1], -1), second.transpose(1, 0, 2).reshape(second.shape[1], -1)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    similarity = a @ b.T
    rows, columns = linear_sum_assignment(-similarity)
    return float(similarity[rows, columns].mean())


def activation_events(h, w, x, *, thresholds, valid_starts, max_error=.95):
    """Suppress nearby peaks and score full templates, retaining model ambiguity."""
    events = []
    length = w.shape[2]
    for k in range(w.shape[1]):
        # Pad so supported events at the first/last timeline samples can be found.
        peaks, _ = find_peaks(np.r_[0, h[k], 0], height=thresholds[k], distance=length)
        for start in peaks - 1:
            if start < 0 or not valid_starts[start] or start + length > x.shape[1]:
                continue
            window, template = x[:, start:start + length], w[:, k, :]
            amplitude = max(0., float(np.sum(window * template) / max(np.sum(template ** 2), 1e-9)))
            error = float(np.sum((window - amplitude * template) ** 2) / max(np.sum(window ** 2), 1e-9))
            if error <= max_error:
                events.append({"factor": k, "start": int(start), "end": int(start + length),
                               "activation": float(h[k, start]), "template_error": error})
    return events
