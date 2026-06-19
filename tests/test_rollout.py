"""
Rollout engine tests — validates shapes, sliding window, action conditioning,
gradient isolation, and memory stability.

Does NOT require trained checkpoints. Constructs fresh untrained model instances
for shape-only verification, same pattern as tests/test_dynamics.py.

Usage:
    python tests/test_rollout.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from tokenizer.video_tokenizer import VideoTokenizer
from lam.latent_action_model import LatentActionModel
from dynamics.dynamics_model import DynamicsModel
from inference.model_loader import build_index_to_latents_fn
from inference.rollout import encode_frame, action_index_to_conditioning, RolloutState

torch.manual_seed(42)

# ── Config matching our defaults ────────────────────────────────
FRAME_SIZE = 64
PATCH_SIZE = 4
P = (FRAME_SIZE // PATCH_SIZE) ** 2  # 256
LATENT_DIM = 5
NUM_BINS = 4
ACTION_DIM = 2
EMBED_DIM = 32
T_CTX = 4
B = 1  # batch size for rollout tests

print("=" * 60)
print("ROLLOUT ENGINE TEST SUITE")
print("=" * 60)

# ── Build fresh untrained models ────────────────────────────────
video_tokenizer = VideoTokenizer(
    frame_size=FRAME_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
    num_heads=8, hidden_dim=128, num_blocks=2,
    latent_dim=LATENT_DIM, num_bins=NUM_BINS, max_frames=16,
)
video_tokenizer.eval()
video_tokenizer.requires_grad_(False)

lam = LatentActionModel(
    embed_dim=EMBED_DIM, num_heads=8, hidden_dim=128, num_blocks=2,
    action_dim=ACTION_DIM, num_bins=NUM_BINS,
)
lam.eval()
lam.requires_grad_(False)

dynamics_model = DynamicsModel(
    embed_dim=EMBED_DIM, num_heads=8, hidden_dim=128, num_blocks=2,
    latent_dim=LATENT_DIM, num_bins=NUM_BINS, action_dim=ACTION_DIM, max_frames=16,
    frame_size=FRAME_SIZE, patch_size=PATCH_SIZE,
)
dynamics_model.eval()
dynamics_model.requires_grad_(False)

video_idx_to_lat = build_index_to_latents_fn(NUM_BINS, LATENT_DIM)
action_idx_to_lat = build_index_to_latents_fn(NUM_BINS, ACTION_DIM)

# ── Test 1: encode_frame shape ──────────────────────────────────
print("\n--- Test 1: encode_frame shape ---")
fake_frame_4d = torch.rand(B, 3, FRAME_SIZE, FRAME_SIZE) * 2 - 1  # [B, C, H, W]
fake_frame_5d = torch.rand(B, 1, 3, FRAME_SIZE, FRAME_SIZE) * 2 - 1  # [B, 1, C, H, W]

z_q_4d = encode_frame(video_tokenizer, fake_frame_4d)
z_q_5d = encode_frame(video_tokenizer, fake_frame_5d)

assert z_q_4d.shape == (B, 1, P, LATENT_DIM), f"Expected {(B, 1, P, LATENT_DIM)}, got {z_q_4d.shape}"
assert z_q_5d.shape == (B, 1, P, LATENT_DIM), f"Expected {(B, 1, P, LATENT_DIM)}, got {z_q_5d.shape}"
print(f"  4D input: {tuple(z_q_4d.shape)} ✓")
print(f"  5D input: {tuple(z_q_5d.shape)} ✓")
print("✅ encode_frame correct")

# ── Test 2: action_index_to_conditioning shape and values ───────
print("\n--- Test 2: action_index_to_conditioning ---")
for action_idx in [0, 5, 15]:
    cond = action_index_to_conditioning(action_idx, action_idx_to_lat, B, torch.device("cpu"))
    assert cond.shape == (B, 1, ACTION_DIM), f"Expected {(B, 1, ACTION_DIM)}, got {cond.shape}"
    # FSQ bin centers are at (bin_idx + 0.5) / num_bins * 2 - 1
    # With num_bins=4, valid centers are: -0.875, -0.375, 0.125, 0.625
    assert cond.min() >= -1.0 and cond.max() <= 1.0, \
        f"Conditioning values out of FSQ range: min={cond.min()}, max={cond.max()}"
    print(f"  Action {action_idx}: shape={tuple(cond.shape)}, values={cond[0, 0].tolist()}")

# Verify specific values: action 0 = (digit0=0, digit1=0) -> (-0.875, -0.875)
cond_0 = action_index_to_conditioning(0, action_idx_to_lat, 1, torch.device("cpu"))
expected_center = (0 + 0.5) / NUM_BINS * 2 - 1  # = -0.75
assert torch.allclose(cond_0[0, 0, 0], torch.tensor(expected_center)), \
    f"Action 0 dim 0 expected {expected_center}, got {cond_0[0, 0, 0].item()}"
print("✅ action_index_to_conditioning correct")

# ── Test 3: RolloutState reset and step ─────────────────────────
print("\n--- Test 3: RolloutState reset + step ---")
rollout = RolloutState(
    video_tokenizer=video_tokenizer,
    dynamics_model=dynamics_model,
    action_index_to_latents_fn=action_idx_to_lat,
    video_index_to_latents_fn=video_idx_to_lat,
    context_length=T_CTX,
    num_maskgit_steps=2,  # low for fast testing
    temperature=0.0,
    schedule_k=5.0,
)

# Seed with fake clip
fake_clip = torch.rand(B, T_CTX, 3, FRAME_SIZE, FRAME_SIZE) * 2 - 1
rollout.reset(fake_clip)

assert rollout.latent_buffer.shape == (B, T_CTX, P, LATENT_DIM), \
    f"Buffer shape after reset: {rollout.latent_buffer.shape}"
assert rollout.action_buffer.shape == (B, T_CTX - 1, ACTION_DIM), \
    f"Action buffer shape after reset: {rollout.action_buffer.shape}"
print(f"  After reset: buffer={tuple(rollout.latent_buffer.shape)}, actions={tuple(rollout.action_buffer.shape)} ✓")

# Take several steps and verify buffer stays at constant length
for i in range(5):
    frame_pixels, frame_np = rollout.step(action_index=i % 16)
    assert rollout.latent_buffer.shape == (B, T_CTX, P, LATENT_DIM), \
        f"Buffer shape changed after step {i+1}: {rollout.latent_buffer.shape}"
    assert rollout.action_buffer.shape == (B, T_CTX - 1, ACTION_DIM), \
        f"Action buffer shape changed after step {i+1}: {rollout.action_buffer.shape}"
    assert frame_pixels.shape == (B, 3, FRAME_SIZE, FRAME_SIZE), \
        f"Frame pixels shape: {frame_pixels.shape}"
    assert frame_np.shape == (FRAME_SIZE, FRAME_SIZE, 3), \
        f"Frame numpy shape: {frame_np.shape}"
    assert 0.0 <= frame_np.min() and frame_np.max() <= 1.0, \
        f"Frame numpy range: [{frame_np.min()}, {frame_np.max()}]"
    print(f"  Step {i+1}: buffer={tuple(rollout.latent_buffer.shape)}, frame={frame_np.shape} ✓")

print("✅ RolloutState reset + step correct, sliding window constant length")

# ── Test 4: No gradient accumulation ────────────────────────────
print("\n--- Test 4: No gradient accumulation (20 steps) ---")

# Verify no tensor in the pipeline has requires_grad=True
for name, p in video_tokenizer.named_parameters():
    assert not p.requires_grad, f"Video Tokenizer param {name} has requires_grad=True!"
for name, p in dynamics_model.named_parameters():
    assert not p.requires_grad, f"Dynamics Model param {name} has requires_grad=True!"

rollout.reset(fake_clip)

for i in range(20):
    _, _ = rollout.step(action_index=i % 16)
    # Verify buffer has no gradient graph
    assert not rollout.latent_buffer.requires_grad, \
        f"latent_buffer acquired requires_grad at step {i+1}!"
    assert not rollout.action_buffer.requires_grad, \
        f"action_buffer acquired requires_grad at step {i+1}!"

print("  20 steps completed, no gradient leaks detected")
print("✅ Gradient isolation verified")

# ── Test 5: build_index_to_latents_fn roundtrip ────────────────
print("\n--- Test 5: index_to_latents roundtrip ---")
from tokenizer.fsq import FiniteScalarQuantizer

# Video codebook roundtrip
vt_quantizer = FiniteScalarQuantizer(LATENT_DIM, NUM_BINS)
for test_idx in [0, 100, 500, 1023]:
    idx_tensor = torch.tensor([test_idx])
    latent = video_idx_to_lat(idx_tensor)  # [1, 5]
    # Get index back
    back_idx = vt_quantizer.get_indices_from_latents(latent)
    assert back_idx.item() == test_idx, \
        f"Video roundtrip failed: {test_idx} -> {latent.tolist()} -> {back_idx.item()}"

# Action codebook roundtrip
act_quantizer = FiniteScalarQuantizer(ACTION_DIM, NUM_BINS)
for test_idx in range(16):
    idx_tensor = torch.tensor([test_idx])
    latent = action_idx_to_lat(idx_tensor)  # [1, 2]
    back_idx = act_quantizer.get_indices_from_latents(latent)
    assert back_idx.item() == test_idx, \
        f"Action roundtrip failed: {test_idx} -> {latent.tolist()} -> {back_idx.item()}"

print("  Video codebook: all tested indices roundtrip correctly")
print("  Action codebook: all 16 indices roundtrip correctly")
print("✅ Index-to-latents roundtrip verified")

print("\n" + "=" * 60)
print("✅ ALL ROLLOUT TESTS PASSED")
print("=" * 60)
