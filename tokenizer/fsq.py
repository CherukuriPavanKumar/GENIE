import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, latent_dim, num_bins):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bins = num_bins

    def forward(self, z):
        z = torch.tanh(z)                                              # (-1, 1)
        z01 = (z + 1) / 2                                                # (0, 1)
        bin_idx = torch.clamp((z01 * self.num_bins).floor(), max=self.num_bins - 1)
        bin_centers = (bin_idx + 0.5) / self.num_bins * 2 - 1            # back to (-1,1), at bin center
        return z + (bin_centers - z).detach()                            # straight-through

    def get_indices_from_latents(self, z_q, dim=-1):
        """Converts quantized values into one integer per patch (base-num_bins digits)."""
        z01 = (z_q + 1) / 2
        bin_idx = torch.clamp((z01 * self.num_bins).floor(), 0, self.num_bins - 1).long()
        multiplier = self.num_bins ** torch.arange(self.latent_dim, device=z_q.device)
        return (bin_idx * multiplier).sum(dim=dim)