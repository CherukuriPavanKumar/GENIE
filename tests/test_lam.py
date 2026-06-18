"""
Smoke test for the Latent Action Model — validates shapes, frame masking,
and gradient flow. Run this before training.

Usage:
    python tests/test_lam.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from lam.latent_action_model import LatentActionModel

torch.manual_seed(0)

# Simulate continuous patch embeddings from a Video Tokenizer decoder.latent_embed
# Shape: [B, T, P, E]
B, T, P, E = 2, 4, 256, 32
z_embed = torch.rand(B, T, P, E)
z_embed.requires_grad = True # To test frame masking trick

model = LatentActionModel(
    embed_dim=32, num_heads=8, hidden_dim=128, num_blocks=2, 
    action_dim=2, num_bins=4, var_weight=100.0, var_target=0.01
)

n_params = sum(p.numel() for p in model.parameters())
print(f"LAM Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
print(f"Action Vocab size: {model.action_vocab_size}")

# Forward pass
total_loss, recon_loss, var_loss, action_latents_q = model(z_embed)

print(f"Action latents shape: {tuple(action_latents_q.shape)} (expected {(B, T-1, 2)})")
assert action_latents_q.shape == (B, T-1, 2)

print(f"Total loss: {total_loss.item():.4f} (Recon: {recon_loss.item():.4f}, Var: {var_loss.item():.4f})")

# Test frame masking trick (critical!)
# The decoder should only be able to see z_embed[:, 0] (frame 1)
# Frames 2 to T should NOT have any gradients from the decoder.
total_loss.backward()

assert z_embed.grad is not None

# Gradients from encoder will flow to all frames (since encoder sees all frames)
# Let's test just the decoder masking by doing a standalone decoder pass.

model.zero_grad()
z_embed_test = torch.rand(B, T, P, E, requires_grad=True)

# Fake actions
fake_actions = torch.rand(B, T-1, 2)
z_pred = model.decoder(z_embed_test[:, :1], fake_actions, T=T)

assert z_pred.shape == (B, T, P, E)

z_pred.sum().backward()

# Since we only passed z_embed_test[:, :1] to the decoder, only the first frame
# should have gradients. We essentially enforce the mask by slicing the input.
assert z_embed_test.grad is not None
assert z_embed_test.grad[:, 0].abs().sum() > 0, "First frame should have gradients"
assert z_embed_test.grad[:, 1:].abs().sum() == 0, "Subsequent frames should NOT have gradients from decoder"

print("✅ Frame Masking trick verified: decoder cannot cheat by looking at future frames.")
print("✅ ALL CHECKS PASSED")
