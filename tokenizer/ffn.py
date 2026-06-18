"""
SwiGLU Feed-Forward Network.

    SwiGLU(x) = W_out · [SiLU(W_gate · x) ⊙ (W_value · x)]

Hidden dim is scaled by 2/3 to keep parameter count comparable to a
standard 2-layer FFN with the same nominal hidden_dim.

Uses pre-norm residual connection with RMSNorm.
"""
import math
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.norms import RMSNorm


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        h = math.floor(2 * hidden_dim / 3)
        self.norm = RMSNorm(embed_dim)
        self.w_gate = nn.Linear(embed_dim, h)
        self.w_value = nn.Linear(embed_dim, h)
        self.w_out = nn.Linear(h, embed_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate = F.silu(self.w_gate(x))
        value = self.w_value(x)
        return residual + self.w_out(gate * value)