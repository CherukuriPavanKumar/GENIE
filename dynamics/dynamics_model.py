"""
Dynamics Model — MaskGIT-style Space-Time Transformer.

The "physics engine" of the world model. Given past video tokens and action
tokens, predicts the next frame's video tokens via iterative parallel decoding.

Training:
    Randomly mask a fraction of tokens across the sequence, predict the masked
    tokens via cross-entropy against the ground-truth token indices.

Inference:
    Start with a fully masked future frame. Over multiple steps, unmask the
    most confident predictions following an exponential schedule until all
    positions are filled.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.positional_encoding import build_spatial_pe, build_temporal_pe
from tokenizer.st_transformer import STTransformer


class DynamicsModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=8, hidden_dim=128, num_blocks=4,
                 latent_dim=5, num_bins=4, action_dim=2, max_frames=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = num_bins ** latent_dim

        # Embed FSQ latent vectors into transformer dimension
        self.latent_embed = nn.Linear(latent_dim, embed_dim)

        # Positional encodings
        # We don't know frame_size here, but we know P = (frame_size/patch_size)^2.
        # We'll register PE buffers lazily or require them as params.
        # For now, we pass num_patches_per_side and compute.
        self.embed_dim = embed_dim
        self.max_frames = max_frames

        # STT backbone with FiLM conditioning on action embeddings
        self.transformer = STTransformer(
            embed_dim, num_heads, hidden_dim, num_blocks,
            causal=True, condition_dim=action_dim,
        )

        # Classification head: predict which of codebook_size tokens goes at each position
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.codebook_size),
        )

        # Learnable mask token in latent space [1, 1, 1, latent_dim]
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)

    def register_pe_buffers(self, num_patches_h, num_patches_w):
        """Register positional encoding buffers. Call once after knowing patch grid size."""
        if not hasattr(self, '_pe_registered') or not self._pe_registered:
            self.register_buffer(
                "spatial_pe",
                build_spatial_pe(num_patches_h, num_patches_w, self.embed_dim),
                persistent=False,
            )
            self.register_buffer(
                "temporal_pe",
                build_temporal_pe(self.max_frames, self.embed_dim),
                persistent=False,
            )
            self._pe_registered = True

    def _apply_masking(self, latents, mask_ratio_min=0.5, mask_ratio_max=1.0):
        """Apply random MaskGIT masking during training.

        Args:
            latents: [B, T, P, L] — FSQ latent vectors (float)
            mask_ratio_min/max: uniform sample range for per-batch mask ratio

        Returns:
            masked_latents: [B, T, P, L] with some positions replaced by mask_token
            mask: [B, T, P] boolean — True where masked
        """
        B, T, P, L = latents.shape
        device = latents.device

        # Sample a random mask ratio for this batch
        mask_ratio = mask_ratio_min + torch.rand((), device=device) * (mask_ratio_max - mask_ratio_min)

        # Create random mask
        mask = torch.rand(B, T, P, device=device) < mask_ratio  # [B, T, P]

        # Guarantee at least one unmasked anchor per (batch, patch) across time
        # This ensures temporal attention always has at least one real token to attend to
        anchor_t = torch.randint(0, T, (B, P), device=device)  # [B, P]
        batch_idx = torch.arange(B, device=device)[:, None]      # [B, 1]
        patch_idx = torch.arange(P, device=device)[None, :]      # [1, P]
        mask[batch_idx, anchor_t, patch_idx] = False

        # Replace masked positions with the learnable mask token
        mask_token = self.mask_token.expand(B, T, P, L)
        masked_latents = torch.where(mask.unsqueeze(-1), mask_token, latents)

        return masked_latents, mask

    def forward(self, latents, conditioning=None, targets=None,
                training=True, mask_ratio_min=0.5, mask_ratio_max=1.0):
        """
        Args:
            latents:      [B, T, P, latent_dim] — FSQ latent vectors (float, not indices)
            conditioning: [B, T, action_dim] — action latent vectors for FiLM
            targets:      [B, T, P] — ground-truth token indices (long) for CE loss
            training:     whether to apply masking and compute loss

        Returns:
            logits:    [B, T, P, codebook_size]
            mask:      [B, T, P] or None — which positions were masked
            loss:      scalar or None — masked cross-entropy loss
        """
        B, T, P, L = latents.shape

        # Lazily register PE buffers based on P
        n = int(math.sqrt(P))
        self.register_pe_buffers(n, n)

        # Apply masking during training
        mask = None
        if training and self.training:
            latents, mask = self._apply_masking(latents, mask_ratio_min, mask_ratio_max)

        # Embed latents → transformer dimension
        x = self.latent_embed(latents)  # [B, T, P, E]

        # Add positional encodings
        x = x + self.spatial_pe           # [1, P, E] broadcasts
        x = x + self.temporal_pe[:, :T]   # [1, T, 1, E] broadcasts

        # Transformer with action conditioning
        x = self.transformer(x, condition=conditioning)  # [B, T, P, E]

        # Classification head
        logits = self.output_head(x)  # [B, T, P, codebook_size]

        # Compute masked cross-entropy loss
        loss = None
        if training and self.training and targets is not None:
            logits_flat = logits.reshape(-1, self.codebook_size)  # [B*T*P, V]
            targets_flat = targets.reshape(-1)                    # [B*T*P]
            mask_flat = mask.reshape(-1).float()                  # [B*T*P]

            per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            # Average loss only over masked positions
            denom = mask_flat.sum().clamp_min(1.0)
            loss = (per_token_loss * mask_flat).sum() / denom

        return logits, mask, loss

    # ── MaskGIT Inference ───────────────────────────────────────

    def _exp_schedule(self, step, total_steps, total_masked, k, device):
        """Exponential unmasking schedule.

        Returns the cumulative number of tokens that should be unmasked by this step.
        """
        if step >= total_steps - 1:
            return torch.tensor(float(total_masked), device=device)
        x = step / max(total_steps, 1)
        k_t = torch.tensor(k, device=device, dtype=torch.float32)
        return total_masked * torch.expm1(k_t * x) / torch.expm1(k_t)

    @torch.no_grad()
    def generate(self, context_latents, num_steps=12, index_to_latents_fn=None,
                 conditioning=None, temperature=0.0, schedule_k=5.0, horizon=1):
        """MaskGIT iterative decoding to predict `horizon` future frames.

        Args:
            context_latents: [B, T_ctx, P, latent_dim] — known context frames (FSQ latents)
            num_steps:       number of iterative unmasking steps
            index_to_latents_fn: callable(indices [B,T,P]) → latents [B,T,P,L]
            conditioning:    [B, T_ctx+horizon, action_dim] — action conditioning for full sequence
            temperature:     0 = argmax, >0 = categorical sampling
            schedule_k:      steepness of exponential schedule
            horizon:         number of future frames to predict

        Returns:
            full_latents: [B, T_ctx + horizon, P, latent_dim] — context + predicted frames
        """
        device = context_latents.device
        dtype = context_latents.dtype
        B, T_ctx, P, L = context_latents.shape

        # Append fully masked future frames
        mask_latents = self.mask_token.to(device, dtype).expand(B, horizon, P, -1)
        input_latents = torch.cat([context_latents, mask_latents], dim=1)  # [B, T_ctx+H, P, L]

        # Track which horizon positions are still masked
        mask = torch.ones(B, horizon, P, dtype=torch.bool, device=device)  # [B, H, P]

        total_masked = horizon * P

        for step in range(num_steps):
            # Forward pass (no masking, no loss)
            logits, _, _ = self.forward(
                input_latents, conditioning=conditioning, training=False,
            )  # [B, T_ctx+H, P, codebook_size]

            # Temperature scaling
            if temperature > 0:
                scaled_logits = logits / temperature
            else:
                scaled_logits = logits

            probs = F.softmax(scaled_logits, dim=-1)  # [B, T_ctx+H, P, V]

            # Get confidence scores and predicted indices
            max_probs, _ = probs.max(dim=-1)  # [B, T_ctx+H, P]

            if temperature > 0:
                B_s, T_s, P_s, V = probs.shape
                predicted_indices = torch.distributions.Categorical(
                    probs=probs.reshape(-1, V)
                ).sample().view(B_s, T_s, P_s)
            else:
                _, predicted_indices = probs.max(dim=-1)  # [B, T_ctx+H, P]

            # Focus on horizon positions only
            horizon_probs = max_probs[:, T_ctx:]  # [B, H, P]

            # Determine how many tokens to unmask at this step
            n_target = self._exp_schedule(step, num_steps, total_masked, schedule_k, device)

            # For each batch element, select the most confident masked positions to unmask
            for b in range(B):
                masked_flat = mask[b].reshape(-1)  # [H*P]
                masked_indices = torch.where(masked_flat)[0]

                if masked_indices.numel() == 0:
                    continue

                # How many to unmask this step
                n_already_unmasked = total_masked - masked_indices.numel()
                n_to_unmask = max(
                    total_masked // 16,  # minimum floor
                    min(
                        int(torch.ceil(n_target).item()) - n_already_unmasked,
                        masked_indices.numel(),
                    ),
                )
                n_to_unmask = max(1, n_to_unmask)

                # Get confidence of currently masked positions
                conf_flat = horizon_probs[b].reshape(-1)[masked_indices]

                if conf_flat.numel() > n_to_unmask:
                    top_idx = torch.topk(conf_flat, n_to_unmask, largest=True).indices
                    sel_flat = masked_indices[top_idx]
                else:
                    sel_flat = masked_indices

                # Map flat indices back to (h, p)
                h_sel = sel_flat // P
                p_sel = sel_flat % P

                # Write predicted latents into the input
                for uh in torch.unique(h_sel):
                    h_mask = (h_sel == uh)
                    p_list = p_sel[h_mask]
                    t_abs = T_ctx + int(uh.item())

                    idx_sel = predicted_indices[b:b+1, t_abs:t_abs+1, p_list]  # [1, 1, n_sel]
                    pred_latents = index_to_latents_fn(idx_sel)                 # [1, 1, n_sel, L]
                    input_latents[b:b+1, t_abs:t_abs+1, p_list] = pred_latents
                    mask[b, int(uh.item()), p_list] = False

            # Early exit if all unmasked
            if not mask.any():
                break

        # Final pass: fill any remaining masked tokens via argmax
        if mask.any():
            logits, _, _ = self.forward(input_latents, conditioning=conditioning, training=False)
            _, final_indices = logits.max(dim=-1)  # [B, T_ctx+H, P]

            for b in range(B):
                h_idx, p_idx = torch.where(mask[b])
                if h_idx.numel() == 0:
                    continue
                for uh in torch.unique(h_idx):
                    h_mask = (h_idx == uh)
                    p_list = p_idx[h_mask]
                    t_abs = T_ctx + int(uh.item())
                    idx_sel = final_indices[b:b+1, t_abs:t_abs+1, p_list]
                    pred_latents = index_to_latents_fn(idx_sel)
                    input_latents[b:b+1, t_abs:t_abs+1, p_list] = pred_latents
                    mask[b, int(uh.item()), p_list] = False

        return input_latents  # [B, T_ctx + horizon, P, L]
