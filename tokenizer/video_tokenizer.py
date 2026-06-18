"""
Video Tokenizer — FSQ-VAE with Space-Time Transformer backbone.

Compresses video clips [B, T, C, H, W] into discrete tokens via:
    Encoder: Patch embed → spatial + temporal PE → STT → linear → FSQ quantize
    Decoder: Linear → spatial + temporal PE → STT → pixel-shuffle unembed

The discrete bottleneck (FSQ) forces compact visual pattern representations
that the downstream Action Tokenizer and Dynamics Model consume.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.norms import RMSNorm
from tokenizer.patch_embed import PatchEmbedding
from tokenizer.positional_encoding import build_spatial_pe, build_temporal_pe
from tokenizer.st_transformer import STTransformer
from tokenizer.fsq import FiniteScalarQuantizer


class VideoTokenizerEncoder(nn.Module):
    def __init__(self, frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                 hidden_dim=128, num_blocks=4, latent_dim=5, max_frames=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, 3, embed_dim)

        n = frame_size // patch_size
        self.register_buffer("spatial_pe", build_spatial_pe(n, n, embed_dim), persistent=False)
        self.register_buffer("temporal_pe", build_temporal_pe(max_frames, embed_dim), persistent=False)

        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.latent_head = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim),
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        B, T = frames.shape[:2]
        x = self.patch_embed(frames)                          # [B, T, P, E]
        x = x + self.spatial_pe                               # [1, P, E] broadcasts over B, T
        x = x + self.temporal_pe[:, :T]                       # [1, T, 1, E] broadcasts over B, P
        x = self.transformer(x)                               # [B, T, P, E]
        return self.latent_head(x)                            # [B, T, P, latent_dim]


class PixelShuffleFrameHead(nn.Module):
    """Reconstructs pixel frames from patch token embeddings.
    Maps each token's embedding to patch_size² × channels pixels,
    then reshuffles into spatial layout.

    [B, T, P, E] → [B, T, C, H, W]
    """

    def __init__(self, embed_dim, patch_size, channels, frame_size):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.n = frame_size // patch_size
        self.to_pixels = nn.Linear(embed_dim, channels * patch_size * patch_size)

    def forward(self, tokens):
        B, T, P, E = tokens.shape
        ps, c, n = self.patch_size, self.channels, self.n

        pixels = self.to_pixels(tokens)                           # [B, T, P, c*ps*ps]
        pixels = pixels.reshape(B, T, n, n, c, ps, ps)
        pixels = pixels.permute(0, 1, 4, 2, 5, 3, 6)            # [B, T, c, row, ps, col, ps]
        pixels = pixels.reshape(B, T, c, n * ps, n * ps)         # [B, T, C, H, W]
        return pixels


class VideoTokenizerDecoder(nn.Module):
    def __init__(self, frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                 hidden_dim=128, num_blocks=4, latent_dim=5, max_frames=16):
        super().__init__()
        self.latent_embed = nn.Linear(latent_dim, embed_dim)

        n = frame_size // patch_size
        self.register_buffer("spatial_pe", build_spatial_pe(n, n, embed_dim), persistent=False)
        self.register_buffer("temporal_pe", build_temporal_pe(max_frames, embed_dim), persistent=False)

        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.frame_head = PixelShuffleFrameHead(embed_dim, patch_size, 3, frame_size)

    def forward(self, latents):
        # latents: [B, T, P, latent_dim]
        B, T = latents.shape[:2]
        x = self.latent_embed(latents)                        # [B, T, P, E]
        x = x + self.spatial_pe                               # spatial position info
        x = x + self.temporal_pe[:, :T]                       # temporal position info
        x = self.transformer(x)                               # [B, T, P, E]
        return self.frame_head(x)                             # [B, T, C, H, W]


class VideoTokenizer(nn.Module):
    """Full Video Tokenizer: Encoder → FSQ → Decoder.

    Forward pass returns (loss, reconstructed_frames) for training.
    Use .tokenize() to get discrete token indices, .detokenize() to decode back.
    """

    def __init__(self, frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                 hidden_dim=128, num_blocks=4, latent_dim=5, num_bins=4, max_frames=16):
        super().__init__()
        self.encoder = VideoTokenizerEncoder(
            frame_size, patch_size, embed_dim, num_heads,
            hidden_dim, num_blocks, latent_dim, max_frames,
        )
        self.decoder = VideoTokenizerDecoder(
            frame_size, patch_size, embed_dim, num_heads,
            hidden_dim, num_blocks, latent_dim, max_frames,
        )
        self.quantizer = FiniteScalarQuantizer(latent_dim, num_bins)
        self.codebook_size = num_bins ** latent_dim

    def forward(self, frames):
        # frames: [B, T, C, H, W], range [-1, 1]
        z = self.encoder(frames)                              # [B, T, P, latent_dim]
        z_q = self.quantizer(z)                               # quantized (STE gradient)
        recon = self.decoder(z_q)                             # [B, T, C, H, W]
        recon_loss = F.mse_loss(recon, frames)
        return recon_loss, recon

    @torch.no_grad()
    def tokenize(self, frames):
        """Encode frames to discrete token indices. [B,T,C,H,W] → [B,T,P]"""
        z = self.encoder(frames)
        z_q = self.quantizer(z)
        return self.quantizer.get_indices_from_latents(z_q)

    def detokenize(self, z_q):
        """Decode quantized latents back to pixel frames. [B,T,P,D] → [B,T,C,H,W]"""
        return self.decoder(z_q)
