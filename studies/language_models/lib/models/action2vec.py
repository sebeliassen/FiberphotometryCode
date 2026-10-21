"""
===============================================================================
Action2Vec: Skip-Gram with Negative Sampling (SGNS) on Behavioral Bouts
===============================================================================
Canonical Paper:
    Mikolov et al., "Distributed Representations of Words and Phrases and their
    Compositionality", NIPS 2013.

Implementation:
    - Trains target (W_in) and context (W_out) embeddings of dimension p=32.
    - Context window: symmetric c = ±3 discrete bouts.
    - Negative sampling: k=5 negative tokens sampled from unigram distribution P_n(s) ∝ count(s)^0.75.
    - Pooling: Mean/TF-IDF pooling over the past W=68.3s window.
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Action2Vec(nn.Module):
    def __init__(self, vocab_size: int = 100, embedding_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Canonical uniform initialization
        initrange = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, target_tokens: torch.Tensor, context_tokens: torch.Tensor, neg_tokens: torch.Tensor):
        """
        Computes SGNS loss:
            L = -log σ(v_c^T v_w) - sum_{i=1}^k log σ(-v_{n_i}^T v_w)
        """
        # (batch_size, embed_dim)
        v_target = self.in_embed(target_tokens)
        v_context = self.out_embed(context_tokens)
        # (batch_size, neg_k, embed_dim)
        v_neg = self.out_embed(neg_tokens)

        # Positive score
        pos_score = torch.sum(v_target * v_context, dim=-1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)

        # Negative score
        neg_score = torch.bmm(v_neg, v_target.unsqueeze(-1)).squeeze(-1)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=-1)

        return torch.mean(pos_loss + neg_loss)

    def get_embeddings(self) -> np.ndarray:
        """Returns the learned input embedding matrix in numpy format."""
        return self.in_embed.weight.detach().cpu().numpy()


def train_action2vec(
    session_bout_sequences: list,
    vocab_size: int = 100,
    embedding_dim: int = 32,
    window_size: int = 3,
    negative_samples: int = 5,
    epochs: int = 15,
    batch_size: int = 4096,
    lr: float = 0.005,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Extracts all (target, context) pairs within session boundaries,
    trains SGNS using Adam optimizer, and returns (vocab_size, embedding_dim) embeddings.
    """
    # 1. Build unigram distribution P_n(s) ∝ count(s)^0.75
    counts = np.zeros(vocab_size, dtype=np.float64)
    pairs = []

    for seq in session_bout_sequences:
        if len(seq) < 2:
            continue
        for s in seq:
            counts[min(s, vocab_size - 1)] += 1.0

        N = len(seq)
        for i, target in enumerate(seq):
            t_id = min(target, vocab_size - 1)
            # Symmetric context window c = ±3
            left = max(0, i - window_size)
            right = min(N, i + window_size + 1)
            for j in range(left, right):
                if i != j:
                    c_id = min(seq[j], vocab_size - 1)
                    pairs.append((t_id, c_id))

    if len(pairs) == 0:
        return np.random.randn(vocab_size, embedding_dim).astype(np.float32)

    counts_pow = np.power(np.maximum(counts, 1.0), 0.75)
    noise_dist = counts_pow / np.sum(counts_pow)
    noise_tensor = torch.tensor(noise_dist, dtype=torch.float32, device=device)

    pairs = np.array(pairs, dtype=np.int64)
    model = Action2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_samples = len(pairs)
    indices = np.arange(n_samples)

    model.train()
    for ep in range(epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            batch_pairs = pairs[batch_idx]

            targets = torch.tensor(batch_pairs[:, 0], dtype=torch.long, device=device)
            contexts = torch.tensor(batch_pairs[:, 1], dtype=torch.long, device=device)

            # Sample k negative tokens
            neg_samples = torch.multinomial(noise_tensor, len(batch_idx) * negative_samples, replacement=True)
            neg_samples = neg_samples.view(len(batch_idx), negative_samples)

            optimizer.zero_grad()
            loss = model(targets, contexts, neg_samples)
            loss.backward()
            optimizer.step()

    return model.get_embeddings()
