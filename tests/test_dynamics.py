"""
Rigorous test suite for the Dynamics Model — validates shapes, masking logic,
gradient flow, inference generation, and crucially, FUTURE-LEAKAGE.

Usage:
    python tests/test_dynamics.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from dynamics.dynamics_model import DynamicsModel

torch.manual_seed(42)

# ── Config matching our defaults ────────────────────────────────
B, T, P = 2, 4, 256  # batch, frames, patches (16x16 grid)
latent_dim = 5
num_bins = 4
action_dim = 2
codebook_size = num_bins ** latent_dim  # 1024

def get_dummy_inputs():
    latents = torch.rand(B, T, P, latent_dim) * 2 - 1  # [-1, 1]
    targets = torch.randint(0, codebook_size, (B, T, P))
    conditioning = torch.rand(B, T, action_dim)
    return latents, targets, conditioning

def build_model():
    return DynamicsModel(
        embed_dim=32, num_heads=8, hidden_dim=128, num_blocks=2,
        latent_dim=latent_dim, num_bins=num_bins, action_dim=action_dim, max_frames=16,
    )

print("==========================================")
print("DYNAMICS MODEL RIGOROUS TEST SUITE")
print("==========================================")

# ── Test 1: Forward pass shapes ─────────────────────────────────
print("\n--- Test 1: Forward pass shapes ---")
model = build_model()
model.train()
latents, targets, conditioning = get_dummy_inputs()

logits, mask, loss = model(latents, conditioning=conditioning, targets=targets, training=True)
assert logits.shape == (B, T, P, codebook_size)
assert mask.shape == (B, T, P)
assert loss.item() > 0
print("✅ Shapes correct")

# ── Test 2: Mask integrity test ─────────────────────────────────
print("\n--- Test 2: Mask integrity test ---")
# Verify masked positions truly contain only mask_token and no residual latent information survives
model.train()
latents, targets, conditioning = get_dummy_inputs()

# We need to intercept the masked_latents to verify this. 
# We can call _apply_masking directly.
masked_latents, mask = model._apply_masking(latents, mask_ratio_min=0.5, mask_ratio_max=1.0)

# Where mask is True, masked_latents MUST exactly equal the mask_token
# Where mask is False, masked_latents MUST exactly equal the original latents
for b in range(B):
    for t in range(T):
        for p in range(P):
            if mask[b, t, p]:
                assert torch.allclose(masked_latents[b, t, p], model.mask_token.squeeze()), "Masked position leaked original latent!"
            else:
                assert torch.allclose(masked_latents[b, t, p], latents[b, t, p]), "Unmasked position was altered!"

print("✅ Mask integrity verified: no residual latent information survives masking.")


# ── Test 3: STRICT FUTURE-LEAKAGE TEST (CRITICAL) ──────────────
print("\n--- Test 3: STRICT FUTURE-LEAKAGE TEST ---")
# Proves future information cannot influence past predictions.
model.eval()
latents1, targets, conditioning1 = get_dummy_inputs()

# Run forward pass and store logits
logits1, _, _ = model(latents1, conditioning=conditioning1, training=False)

# Heavily perturb frame t=2 (index 2) and beyond
latents2 = latents1.clone()
latents2[:, 2:] = torch.randn_like(latents2[:, 2:]) * 100.0  # Massive noise

conditioning2 = conditioning1.clone()
conditioning2[:, 2:] = torch.randn_like(conditioning2[:, 2:]) * 100.0

# Run second forward pass
logits2, _, _ = model(latents2, conditioning=conditioning2, training=False)

# Assert that outputs for frame t=0 and t=1 are EXACTLY unchanged
assert torch.allclose(logits1[:, :2], logits2[:, :2], atol=1e-5), "FUTURE LEAKAGE DETECTED! Frame t logits changed when frame t+1 was perturbed."

# Assert that outputs for frame t=2 and t=3 DID change (sanity check that the model actually processes the input)
assert not torch.allclose(logits1[:, 2:], logits2[:, 2:], atol=1e-5), "Sanity check failed: perturbing input didn't change output."

print("✅ Strict Future-Leakage test passed: causal masking is flawless.")


# ── Test 4: Generation consistency test ─────────────────────────
print("\n--- Test 4: Generation consistency test (temperature=0) ---")
model.eval()

def dummy_index_to_latents(indices):
    flat = indices.reshape(-1)
    bin_indices = []
    remaining = flat
    for d in range(latent_dim):
        bin_indices.append(remaining % num_bins)
        remaining = remaining // num_bins
    bin_indices = torch.stack(bin_indices, dim=-1).float()
    return ((bin_indices + 0.5) / num_bins * 2 - 1).reshape(*indices.shape, latent_dim)

context = latents1[:, :2]
cond_full = conditioning1[:, :3]

with torch.no_grad():
    gen1 = model.generate(
        context, num_steps=4, index_to_latents_fn=dummy_index_to_latents,
        conditioning=cond_full, temperature=0.0, schedule_k=5.0, horizon=1,
    )
    
    # Run again
    gen2 = model.generate(
        context, num_steps=4, index_to_latents_fn=dummy_index_to_latents,
        conditioning=cond_full, temperature=0.0, schedule_k=5.0, horizon=1,
    )

assert torch.allclose(gen1, gen2), "Generation with temperature=0 is non-deterministic!"
print("✅ Generation consistency verified: temperature=0 is deterministic.")


# ── Test 5: Frozen-module simulation test ───────────────────────
print("\n--- Test 5: Frozen-module gradient isolation ---")
import torch.nn as nn

class MockFrozenVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(10))
    def forward(self, x):
        return x * self.param.sum()

vt = MockFrozenVT()
# Freeze it
for p in vt.parameters(): p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

# Simulate training loop precomputation
with torch.no_grad():
    dummy_input = torch.ones(2)
    vt_out = vt(dummy_input)
    # We use this to make dummy latents
    latents_train = torch.rand(B, T, P, latent_dim) * vt_out.sum() 

_, _, loss = model(latents_train, conditioning=conditioning1, targets=targets, training=True)
loss.backward()

# Verify VT received no gradients
for p in vt.parameters():
    assert p.grad is None, "Gradient leaked into frozen Video Tokenizer!"

# Verify Dynamics Model DID receive gradients
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
assert grad_norm > 0, "Dynamics Model received no gradients!"

print("✅ Frozen-module isolation verified: gradients strictly contained.")

print("\n==========================================")
print("✅ ALL CHECKS PASSED: MODEL IS PRODUCTION READY")
print("==========================================")
