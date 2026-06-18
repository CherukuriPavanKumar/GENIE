"""
Spatial and Temporal multi-head attention for the Space-Time Transformer.

Spatial attention:  each patch attends to all patches within the SAME frame.
Temporal attention: each patch attends to the SAME spatial position across frames
                    (causal — only looks at current + past frames).

Both use pre-norm residual connections with RMSNorm.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.norms import RMSNorm


class SpatialAttention(nn.Module):
    """Patches within a single frame attend to each other.
    Input: [B, T, P, E] → Output: [B, T, P, E]
    Internally reshapes to (B*T, P, H, D) so frames are independent."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm = RMSNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, P, E]
        B, T, P, E = x.shape
        H, D = self.num_heads, self.head_dim

        residual = x
        x = self.norm(x)

        # reshape to (B*T, P, E) — each frame is an independent sequence
        q = self.q_proj(x).reshape(B * T, P, H, D).transpose(1, 2)   # [B*T, H, P, D]
        k = self.k_proj(x).reshape(B * T, P, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B * T, P, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)                           # [B*T, H, P, D]

        attn_out = attn_out.transpose(1, 2).reshape(B, T, P, E)
        attn_out = self.out_proj(attn_out)
        return residual + attn_out


class TemporalAttention(nn.Module):
    """Same spatial position attends across time steps.
    Input: [B, T, P, E] → Output: [B, T, P, E]
    Internally reshapes to (B*P, T, H, D) so each spatial slot is independent.
    Causal by default — each frame can only attend to itself and earlier frames."""

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.norm = RMSNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, P, E]
        B, T, P, E = x.shape
        H, D = self.num_heads, self.head_dim

        residual = x
        x = self.norm(x)

        # reshape to (B*P, T, E) — each spatial slot is an independent sequence
        x_t = x.permute(0, 2, 1, 3).reshape(B * P, T, E)

        q = self.q_proj(x_t).reshape(B * P, T, H, D).transpose(1, 2)   # [B*P, H, T, D]
        k = self.k_proj(x_t).reshape(B * P, T, H, D).transpose(1, 2)
        v = self.v_proj(x_t).reshape(B * P, T, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)                             # [B*P, H, T, D]

        attn_out = attn_out.transpose(1, 2).reshape(B, P, T, E)
        attn_out = attn_out.permute(0, 2, 1, 3)                        # back to [B, T, P, E]
        attn_out = self.out_proj(attn_out)
        return residual + attn_out