import math
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        h = math.floor(2 * hidden_dim / 3)
        self.w_value = nn.Linear(embed_dim, h)
        self.w_gate = nn.Linear(embed_dim, h)
        self.w_out = nn.Linear(h, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        v = F.silu(self.w_value(x))
        g = self.w_gate(x)
        out = self.w_out(v * g)
        return self.norm(x + out)