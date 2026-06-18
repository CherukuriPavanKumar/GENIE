"""
Space-Time Transformer — the backbone of the Video Tokenizer, Action Tokenizer,
and Dynamics Model.

Each STTransformerBlock follows:
    Input → RMSNorm → Spatial Attention → RMSNorm → Temporal Attention → RMSNorm → SwiGLU FFN → Output

All sub-layers use pre-norm residual connections (norm is inside each component).

STTransformer stacks N such blocks.
"""
import torch.nn as nn

from tokenizer.attention import SpatialAttention, TemporalAttention
from tokenizer.ffn import SwiGLUFFN


class STTransformerBlock(nn.Module):
    """Single Space-Time Transformer block.

    Spatial Attention → Temporal Attention → SwiGLU FFN
    Each sub-layer has its own pre-norm and residual connection internally.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, causal: bool = True):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal=causal)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, P, E]
        x = self.spatial_attn(x)     # patches attend within each frame
        x = self.temporal_attn(x)    # same position attends across time
        x = self.ffn(x)              # channel mixing
        return x


class STTransformer(nn.Module):
    """Stack of N Space-Time Transformer blocks.

    Input: [B, T, P, E] → Output: [B, T, P, E]
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int,
                 num_blocks: int, causal: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            STTransformerBlock(embed_dim, num_heads, hidden_dim, causal=causal)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: [B, T, P, E]
        for block in self.blocks:
            x = block(x)
        return x