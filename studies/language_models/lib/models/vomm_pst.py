"""
===============================================================================
Variable-Order Markov Model: Probabilistic Suffix Tree (PST)
===============================================================================
Canonical Papers:
    - Ron, Singer, & Tishby, "The Power of Amnesia: Learning Probabilistic
      Automata with Variable Memory Length", Machine Learning 1996.
    - Bejerano & Yona, "Variations on probabilistic suffix trees: statistical
      modeling and prediction of protein families", RECOMB 2001.

Implementation:
    - Builds a compact suffix tree of maximum depth D_max (e.g. 5 bouts).
    - Context node s is retained if empirical frequency P(s) >= P_min and
      P(k|s) / P(k|parent(s)) >= r for at least one token k (predictive divergence).
    - Prediction: Traversing the suffix tree to deepest matching context node
      with smoothed conditional probabilities.
===============================================================================
"""

from collections import defaultdict

import numpy as np


class PSTNode:
    def __init__(self, context: tuple, vocab_size: int = 100):
        self.context = context  # trailing string (..., s_{-2}, s_{-1})
        self.vocab_size = vocab_size
        self.counts = np.zeros(vocab_size, dtype=np.float64)
        self.total_count = 0.0
        self.prob = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float32)
        self.children = {}  # prefix symbol -> child PSTNode

    def update_counts(self, next_symbol: int):
        self.counts[next_symbol] += 1.0
        self.total_count += 1.0

    def compute_smoothed_prob(self, smoothing_gamma: float = 0.001):
        """Krichevsky-Trofimov / Laplace smoothing."""
        denom = self.total_count + smoothing_gamma * self.vocab_size
        if denom > 0:
            self.prob = ((self.counts + smoothing_gamma) / denom).astype(np.float32)
        else:
            self.prob = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float32)


class ProbabilisticSuffixTree:
    def __init__(
        self,
        vocab_size: int = 100,
        max_depth: int = 5,
        min_count: int = 5,
        divergence_ratio: float = 1.05,
        smoothing_gamma: float = 0.001
    ):
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.min_count = min_count
        self.divergence_ratio = divergence_ratio
        self.smoothing_gamma = smoothing_gamma
        self.root = PSTNode(context=(), vocab_size=vocab_size)

    def fit(self, sequences: list):
        """Builds and prunes the PST from training sequences."""
        # 1. Count n-grams up to max_depth + 1
        ngram_counts = defaultdict(lambda: np.zeros(self.vocab_size, dtype=np.float64))

        for seq in sequences:
            N = len(seq)
            if N < 2:
                continue
            for i in range(N - 1):
                next_sym = min(seq[i + 1], self.vocab_size - 1)
                # Suffixes of length 0 to max_depth
                for d in range(0, min(self.max_depth + 1, i + 1)):
                    context = tuple(seq[i - d : i + 1])
                    ngram_counts[context][next_sym] += 1.0

        # Root node
        root_counts = np.zeros(self.vocab_size, dtype=np.float64)
        for seq in sequences:
            for s in seq:
                root_counts[min(s, self.vocab_size - 1)] += 1.0
        self.root.counts = root_counts
        self.root.total_count = float(np.sum(root_counts))
        self.root.compute_smoothed_prob(self.smoothing_gamma)

        # 2. Add contexts that meet frequency and divergence criteria
        nodes = {(): self.root}

        # Sort contexts by increasing length
        sorted_contexts = sorted(ngram_counts.keys(), key=lambda x: len(x))

        for ctx in sorted_contexts:
            if len(ctx) > self.max_depth:
                continue
            counts = ngram_counts[ctx]
            total = float(np.sum(counts))
            if total < self.min_count:
                continue

            parent_ctx = ctx[1:]  # suffix without leftmost token
            if parent_ctx in nodes:
                parent_node = nodes[parent_ctx]
                # Check predictive divergence ratio r
                p_ctx = (counts + self.smoothing_gamma) / (total + self.smoothing_gamma * self.vocab_size)
                p_parent = parent_node.prob

                divergence = np.max(p_ctx / np.maximum(p_parent, 1e-6))
                if divergence >= self.divergence_ratio or np.min(p_ctx / np.maximum(p_parent, 1e-6)) <= (1.0 / self.divergence_ratio):
                    node = PSTNode(context=ctx, vocab_size=self.vocab_size)
                    node.counts = counts
                    node.total_count = total
                    node.compute_smoothed_prob(self.smoothing_gamma)
                    nodes[ctx] = node
                    # Link to parent
                    parent_node.children[ctx[0]] = node

    def predict_distribution(self, context_sequence: list) -> np.ndarray:
        """
        Traverses tree from longest matching suffix to root, returning smoothed
        conditional next-token distribution.
        """
        if len(context_sequence) == 0:
            return self.root.prob

        # Match longest suffix
        cur_ctx = tuple(context_sequence[-self.max_depth:])
        for d in range(len(cur_ctx), 0, -1):
            sub_ctx = cur_ctx[-d:]
            # Search from root
            node = self.root
            matched = True
            for sym in sub_ctx:
                if sym in node.children:
                    node = node.children[sym]
                else:
                    matched = False
                    break
            if matched:
                return node.prob

        return self.root.prob
