from typing import Dict, List

import numpy as np


class Behavior2Vec:
    """
    Implements behavioral embeddings (Word2Vec wrapper or custom factorizer)
    to map syllable sequences into continuous vector representations.
    """
    def __init__(self, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.embeddings = None

    def fit(self, sentences: List[List[str]], postural_priors: Dict[str, np.ndarray] = None):
        """
        Trains embeddings on the sentences.
        If postural_priors is provided, maps each syllable initial weight to its mean PCA centroid.
        """
        unique_sylls = sorted(list(set([syll for sent in sentences for syll in sent])))
        self.vocab = {syll: idx for idx, syll in enumerate(unique_sylls)}

        # Initialize embeddings
        V = len(unique_sylls)
        self.embeddings = np.random.randn(V, self.embedding_dim) * 0.1

        if postural_priors is not None:
            # Map/inject continuous pose dimensions into embeddings initial state
            for syll, vector in postural_priors.items():
                if syll in self.vocab:
                    idx = self.vocab[syll]
                    # Project/pad prior vector to match embedding_dim
                    self.embeddings[idx, :len(vector)] = vector[:self.embedding_dim]

        # Placeholder for SGD/Word2Vec optimization
        print(f"Initialized {V} syllable embeddings with dimension {self.embedding_dim}.")

    def get_vector(self, syllable: str) -> np.ndarray:
        if syllable in self.vocab:
            return self.embeddings[self.vocab[syllable]]
        return np.zeros(self.embedding_dim)
