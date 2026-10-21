"""Backoff sequence-history prediction on explicitly separated bout segments."""

from collections import Counter, defaultdict

import numpy as np


class HistoryPredictor:
    """Dirichlet-smoothed conditional token model, with observed-history backoff."""

    def __init__(self, vocabulary, max_history=5, strength=10.):
        self.vocabulary, self.max_history, self.strength = vocabulary, max_history, strength

    def fit(self, sequences):
        self.counts = [defaultdict(Counter) for _ in range(self.max_history + 1)]
        for seq in sequences:
            for i, token in enumerate(seq):
                for history in range(min(i, self.max_history) + 1):
                    context = tuple(seq[i - history:i])
                    self.counts[history][context][int(token)] += 1
        return self

    def score(self, sequences, history):
        losses, supported = [], []
        for sequence in sequences:
            for i in range(self.max_history, len(sequence)):
                token = int(sequence[i])
                base = self.counts[0][()]
                probability = (base.get(token, 0) + 1) / (sum(base.values()) + self.vocabulary)
                highest_supported = 0
                for length in range(1, history + 1):
                    count = self.counts[length].get(tuple(sequence[i - length:i]))
                    if count:
                        n = sum(count.values())
                        probability = (count.get(token, 0) + self.strength * probability) / (n + self.strength)
                        highest_supported = length
                losses.append(-np.log2(max(probability, 1e-12)))
                supported.append(highest_supported == history)
        return {"cross_entropy_bits": float(np.mean(losses)) if losses else np.nan,
                "n_predictions": len(losses), "full_history_support_fraction": float(np.mean(supported)) if supported else np.nan}
