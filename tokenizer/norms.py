"""
RMSNorm — simpler and cheaper than LayerNorm.
Divides by root-mean-square of activations, no mean-centering.

    RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

Used as the default normalisation throughout the Space-Time Transformer.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
