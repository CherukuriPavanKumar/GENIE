import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from tokenizer.video_tokenizer import VideoTokenizer

torch.manual_seed(0)
B, T, C, H, W = 2, 4, 3, 64, 64
frames = torch.rand(B, T, C, H, W) * 2 - 1  # fake data, same [-1,1] range as the real pipeline

model = VideoTokenizer(frame_size=64, patch_size=4, embed_dim=32, num_heads=8,
                        hidden_dim=128, num_blocks=4, latent_dim=5, num_bins=4)

print(f"Codebook size: {model.codebook_size}")

recon_loss, recon = model(frames)
print(f"recon shape: {tuple(recon.shape)} (expected {(B, T, C, H, W)})")
assert recon.shape == frames.shape

indices = model.tokenize(frames)
print(f"token indices shape: {tuple(indices.shape)} (expected {(B, T, 256)})")
assert indices.shape == (B, T, 256)
assert indices.min() >= 0 and indices.max() < model.codebook_size

recon_loss.backward()
print("backward() ran cleanly -- gradients flow through the whole model")
print("ALL CHECKS PASSED")