"""
===============================================================================
Causal Action Transformer (Action-GPT) with Robust Causal Self-Attention
===============================================================================
Canonical Papers:
    - Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019.
    - Vaswani et al., "Attention Is All You Need", NIPS 2017.

Implementation:
    - Token Embedding (d_model=64) + Continuous Duration Projection MLP (64D).
    - Learned 1D Positional Encodings.
    - 2 Causal Multi-Head Self-Attention layers with robust lower-triangular attention mask.
    - Pre-LayerNorm architecture with GELU feedforward expansion (256D).
    - Output projection head predicting future action distribution on simplex.
===============================================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, L, d_model)
        mask: (B, L) boolean mask (True for real tokens, False for pad)
        """
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, L, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, L, L)

        # 1. Causal lower-triangular mask
        causal_mask = torch.triu(torch.full((L, L), -1e4, device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # 2. Key padding mask
        if mask is not None:
            pad_mask = (~mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(pad_mask, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CausalActionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.dur_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, durations: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        tokens: (B, L)
        durations: (B, L) in seconds
        mask: (B, L) boolean mask
        """
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)

        dur_log = torch.log1p(torch.clamp(durations, min=0.0)).unsqueeze(-1)
        x = self.token_embed(tokens) + self.dur_proj(dur_log) + self.pos_embed(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_ln(x)

        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pool = torch.sum(x * mask_f, dim=1) / torch.clamp(torch.sum(mask_f, dim=1), min=1.0)
        else:
            pool = torch.mean(x, dim=1)

        logits = self.head(pool)
        return F.softmax(logits, dim=-1)
