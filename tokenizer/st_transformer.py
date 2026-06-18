"""
Space-Time Transformer — the backbone of the Video Tokenizer, Action Tokenizer,
and Dynamics Model.

Each STTransformerBlock follows:
    Input → RMSNorm → Spatial Attention → RMSNorm → Temporal Attention → RMSNorm → SwiGLU FFN → Output

All sub-layers use pre-norm residual connections (norm is inside each component).
If condition_dim is provided, FiLMRMSNorm is used for the FFN norm to inject action conditioning.

STTransformer stacks N such blocks.
"""
import torch
import torch.nn as nn

from tokenizer.attention import SpatialAttention, TemporalAttention
from tokenizer.ffn import SwiGLUFFN
from tokenizer.norms import FiLMRMSNorm


class STTransformerBlock(nn.Module):
    """Single Space-Time Transformer block.

    Spatial Attention → Temporal Attention → SwiGLU FFN
    Each sub-layer has its own pre-norm and residual connection internally.
    If condition_dim is provided, we use FiLM on the FFN to inject actions.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, causal: bool = True, condition_dim: int = None):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal=causal)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim)
        
        self.condition_dim = condition_dim
        if condition_dim is not None:
            # We override the norm in FFN with a FiLM norm
            self.ffn.norm = FiLMRMSNorm(embed_dim, condition_dim)

    def forward(self, x, condition=None):
        # x: [B, T, P, E]
        x = self.spatial_attn(x)     # patches attend within each frame
        x = self.temporal_attn(x)    # same position attends across time
        
        # In FFN, if we have a FiLM norm, we must pass the condition.
        # But SwiGLUFFN forward doesn't accept condition directly.
        # We need to manually do the residual + norm inside SwiGLUFFN if condition is provided,
        # OR we just update SwiGLUFFN to optionally take condition.
        # Let's handle it by overriding the forward pass logic of ffn here or update ffn.py
        # Wait, it's cleaner to update SwiGLUFFN in ffn.py to accept condition.
        
        # Let's pass condition down to FFN
        if condition is not None:
            # We must apply condition to FFN
            residual = x
            x_normed = self.ffn.norm(x, condition)
            gate = torch.nn.functional.silu(self.ffn.w_gate(x_normed))
            value = self.ffn.w_value(x_normed)
            x = residual + self.ffn.w_out(gate * value)
        else:
            x = self.ffn(x)              # channel mixing
        return x


class STTransformer(nn.Module):
    """Stack of N Space-Time Transformer blocks.

    Input: [B, T, P, E] → Output: [B, T, P, E]
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int,
                 num_blocks: int, causal: bool = True, condition_dim: int = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            STTransformerBlock(embed_dim, num_heads, hidden_dim, causal=causal, condition_dim=condition_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x, condition=None):
        # x: [B, T, P, E]
        for block in self.blocks:
            x = block(x, condition)
        return x