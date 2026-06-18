import math
import torch


def build_spatial_pe(num_patches_h, num_patches_w, embed_dim):
    """2D sin/cos positional encoding for a patch grid. Half the embedding
    dims encode row position, half encode column position. Returns [1, P, E]."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    dim_per_axis = embed_dim // 2

    def sincos_1d(positions, dim):
        freqs = torch.exp(-math.log(10000) * torch.arange(0, dim, 2).float() / dim)
        args = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    rows = torch.arange(num_patches_h).repeat_interleave(num_patches_w)
    cols = torch.arange(num_patches_w).repeat(num_patches_h)

    row_pe = sincos_1d(rows, dim_per_axis)
    col_pe = sincos_1d(cols, dim_per_axis)

    pe = torch.cat([row_pe, col_pe], dim=-1)
    return pe.unsqueeze(0)  # [1, P, E]


def build_temporal_pe(num_frames, embed_dim):
    """1D sin/cos positional encoding across timesteps. Returns [1, T, 1, E]
    so it broadcasts across every patch within a frame."""
    assert embed_dim % 2 == 0, "embed_dim must be even"
    positions = torch.arange(num_frames).float()
    freqs = torch.exp(-math.log(10000) * torch.arange(0, embed_dim, 2).float() / embed_dim)
    args = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return pe.unsqueeze(0).unsqueeze(2)  # [1, T, 1, E]