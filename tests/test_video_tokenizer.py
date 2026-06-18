"""
Smoke test for the Video Tokenizer — validates shapes, gradient flow,
and codebook utilization on synthetic data. Run this BEFORE training.

Usage:
    python tests/test_video_tokenizer.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from tokenizer.video_tokenizer import VideoTokenizer

torch.manual_seed(0)
B, T, C, H, W = 2, 4, 3, 64, 64
frames = torch.rand(B, T, C, H, W) * 2 - 1  # fake data, same [-1,1] range as real pipeline

model = VideoTokenizer(
    frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
    hidden_dim=128, num_blocks=4, latent_dim=5, num_bins=4, max_frames=16,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
print(f"Codebook size: {model.codebook_size}")

# Forward pass
recon_loss, recon = model(frames)
print(f"Recon shape: {tuple(recon.shape)} (expected {(B, T, C, H, W)})")
assert recon.shape == frames.shape, f"Shape mismatch: {recon.shape} != {frames.shape}"

# Tokenization
indices = model.tokenize(frames)
P = (H // 4) ** 2  # 256 patches
print(f"Token indices shape: {tuple(indices.shape)} (expected {(B, T, P)})")
assert indices.shape == (B, T, P)
assert indices.min() >= 0 and indices.max() < model.codebook_size

# Codebook utilization
n_unique = indices.unique().numel()
print(f"Codebook utilization: {n_unique}/{model.codebook_size} "
      f"({100 * n_unique / model.codebook_size:.1f}%)")

# Gradient flow
recon_loss.backward()
print("backward() ran cleanly — gradients flow through STE and entire model")

# Check encoder gradients specifically (STE can silently block them)
enc_grad_norm = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
print(f"Encoder gradient norm: {enc_grad_norm:.4f} (should be > 0)")
assert enc_grad_norm > 0, "Encoder gradients are zero — STE may be broken!"

print("\n✅ ALL CHECKS PASSED")