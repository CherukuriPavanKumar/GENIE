import torch.nn as nn
import torch.nn.functional as F

from tokenizer.patch_embed import PatchEmbedding
from tokenizer.positional_encoding import build_spatial_pe
from tokenizer.st_transformer import STTransformer
from tokenizer.fsq import FiniteScalarQuantizer


class VideoTokenizerEncoder(nn.Module):
    def __init__(self, frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                 hidden_dim=128, num_blocks=4, latent_dim=5):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, 3, embed_dim)
        n = frame_size // patch_size
        self.register_buffer("spatial_pe", build_spatial_pe(n, n, embed_dim), persistent=False)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.latent_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, latent_dim))

    def forward(self, frames):
        x = self.patch_embed(frames)                # [B,T,P,E]
        x = x + self.spatial_pe.unsqueeze(1)
        x = self.transformer(x)
        return self.latent_head(x)                    # [B,T,P,latent_dim]


class PixelShuffleFrameHead(nn.Module):
    def __init__(self, embed_dim, patch_size, channels, frame_size):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.n = frame_size // patch_size
        self.to_pixels = nn.Linear(embed_dim, channels * patch_size * patch_size)

    def forward(self, tokens):
        B, T, P, E = tokens.shape
        ps, c, n = self.patch_size, self.channels, self.n

        pixels = self.to_pixels(tokens)                          # [B,T,P,c*ps*ps]
        pixels = pixels.reshape(B, T, n, n, c, ps, ps)
        pixels = pixels.permute(0, 1, 4, 2, 5, 3, 6)               # [B,T,c,row,ps,col,ps]
        pixels = pixels.reshape(B, T, c, n * ps, n * ps)            # [B,T,C,H,W]
        return pixels


class VideoTokenizerDecoder(nn.Module):
    def __init__(self, frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                 hidden_dim=128, num_blocks=4, latent_dim=5):
        super().__init__()
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        n = frame_size // patch_size
        self.register_buffer("spatial_pe", build_spatial_pe(n, n, embed_dim), persistent=False)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.frame_head = PixelShuffleFrameHead(embed_dim, patch_size, 3, frame_size)

    def forward(self, latents):
        x = self.latent_embed(latents)               # [B,T,P,E]
        x = x + self.spatial_pe.unsqueeze(1)
        x = self.transformer(x)
        return self.frame_head(x)                      # [B,T,C,H,W]


class VideoTokenizer(nn.Module):
    def __init__(self, frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                 hidden_dim=128, num_blocks=4, latent_dim=5, num_bins=4):
        super().__init__()
        self.encoder = VideoTokenizerEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.decoder = VideoTokenizerDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.quantizer = FiniteScalarQuantizer(latent_dim, num_bins)
        self.codebook_size = num_bins ** latent_dim

    def forward(self, frames):
        z = self.encoder(frames)
        z_q = self.quantizer(z)
        recon = self.decoder(z_q)
        recon_loss = F.smooth_l1_loss(recon, frames)
        return recon_loss, recon

    def tokenize(self, frames):
        z = self.encoder(frames)
        z_q = self.quantizer(z)
        return self.quantizer.get_indices_from_latents(z_q)

    def detokenize(self, z_q):
        return self.decoder(z_q)

