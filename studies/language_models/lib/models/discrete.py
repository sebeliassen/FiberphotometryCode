from typing import Dict, List, Tuple  # noqa: F401 - retain historical module attributes

import numpy as np  # noqa: F401 - retain historical module attributes


class DiscreteLanguageModel:
    """
    N-gram discrete language model with Laplace or Kneser-Ney smoothing.
    Quantifies sequence transition rules and calculates perplexity.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.counts = {}
        self.context_counts = {}

    def fit(self, sentences: List[List[str]]):
        """Build counts for N-grams and context."""
        for sentence in sentences:
            if len(sentence) < self.n:
                continue
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i+self.n])
                context = ngram[:-1]
                self.counts[ngram] = self.counts.get(ngram, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1

        print(f"Fitted discrete {self.n}-gram model with {len(self.counts)} unique transitions.")

    def get_probability(self, context: Tuple[str, ...], target: str) -> float:
        """ Laplace smoothing transition probability """
        context = tuple(context)
        ngram = context + (target,)
        num = self.counts.get(ngram, 0) + 1.0
        den = self.context_counts.get(context, 0) + 100.0  # vocab size pseudocount
        return num / den
