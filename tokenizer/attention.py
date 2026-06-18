import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, P, E = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B * T, P, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B * T, P, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B * T, P, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, P, E)
        attn_out = self.out_proj(attn_out)
        return self.norm(x + attn_out)


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, P, E = x.shape
        H, D = self.num_heads, self.head_dim

        x_t = x.permute(0, 2, 1, 3)  # [B, P, T, E] -- group by patch position, not frame

        q = self.q_proj(x_t).reshape(B * P, T, H, D).transpose(1, 2)
        k = self.k_proj(x_t).reshape(B * P, T, H, D).transpose(1, 2)
        v = self.v_proj(x_t).reshape(B * P, T, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, P, T, E)
        attn_out = attn_out.permute(0, 2, 1, 3)  # back to [B, T, P, E]
        attn_out = self.out_proj(attn_out)
        return self.norm(x + attn_out)