"""
RMSNorm — simpler and cheaper than LayerNorm.
Divides by root-mean-square of activations, no mean-centering.

    RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

Used as the default normalisation throughout the Space-Time Transformer.

FiLMRMSNorm modulates the RMSNorm based on an action condition, used by LAM.
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


class FiLMRMSNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.film_gen = nn.Linear(condition_dim, 2 * dim)
        # Initialize to identity transform (scale=1 (which is 1+0), shift=0)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, E)
        # condition: (B, T, E) or similar broadcastable shape, usually we need to match seq lengths.
        # Actually in LAM, the condition has shape (B, T, condition_dim) but needs to broadcast over P.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        
        # film_params: (B, T, 2*E)
        film_params = self.film_gen(condition)
        # We need to reshape to (B, T, 1, 2*E) to broadcast over P patches
        if film_params.dim() == 3:
            film_params = film_params.unsqueeze(2)
        elif film_params.dim() == 2:
            film_params = film_params.unsqueeze(1).unsqueeze(2)

        gamma, beta = film_params.chunk(2, dim=-1)
        # gamma is a scale multiplier, we do (1 + gamma) to center at 1
        return x_normed * (1 + gamma) + beta
