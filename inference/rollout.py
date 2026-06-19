"""
Rollout Engine — core stepping logic for autoregressive world generation.

Separate from any UI/keyboard concerns. This module handles:
  - Encoding real frames into the z_q latent space
  - Converting integer action indices to continuous FiLM conditioning vectors
  - Maintaining a sliding context window of recent latents
  - Stepping the dynamics model to predict the next frame

All public functions use @torch.no_grad() — nothing here should ever build
a computational graph or accumulate gradients.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np


@torch.no_grad()
def encode_frame(video_tokenizer, frame):
    """Encode a raw pixel frame into FSQ-quantized latents.

    Args:
        video_tokenizer: frozen VideoTokenizer instance
        frame: [B, 1, C, H, W] or [B, C, H, W] — pixel frame in [-1, 1]

    Returns:
        z_q: [B, 1, P, 5] — FSQ-quantized latent vectors
    """
    if frame.dim() == 4:
        frame = frame.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
    z = video_tokenizer.encoder(frame)       # [B, 1, P, 5]
    z_q = video_tokenizer.quantizer(z)       # [B, 1, P, 5]
    return z_q


@torch.no_grad()
def action_index_to_conditioning(action_index, action_index_to_latents_fn, batch_size, device):
    """Convert an integer action index (0-15) to a continuous conditioning vector.

    This is the critical bridge between the user's discrete key press and the
    continuous [B, 1, action_dim=2] vector that FiLM conditioning expects.

    The conversion uses the exact same FSQ bin-center math as the LAM's quantizer:
        index -> decompose into base-4 digits -> bin_center = (digit + 0.5) / 4 * 2 - 1

    Args:
        action_index: integer in [0, 15], or tensor of shape [B]
        action_index_to_latents_fn: callable built by build_index_to_latents_fn(num_bins=4, latent_dim=2)
        batch_size: B
        device: torch device

    Returns:
        conditioning: [B, 1, 2] — continuous FSQ bin-center vector for this action
    """
    if isinstance(action_index, int):
        indices = torch.full((batch_size,), action_index, dtype=torch.long, device=device)  # [B]
    else:
        indices = action_index.to(device)  # [B]

    # action_index_to_latents_fn maps [B] -> [B, action_dim=2]
    latents = action_index_to_latents_fn(indices)  # [B, 2]
    return latents.unsqueeze(1)  # [B, 1, 2]


class RolloutState:
    """Maintains the sliding context window for autoregressive generation.

    The dynamics model was trained on fixed-length sequences (seq_len=4 by default).
    During inference, we maintain a buffer of the most recent T_ctx frames.
    After each step, we drop the oldest frame and append the newly generated one,
    keeping the buffer at constant length T_ctx. This is necessary because:
      1. The dynamics model has a fixed max_frames it was trained with — feeding an
         ever-growing sequence would eventually exceed that limit.
      2. Even within that limit, the model was trained on sequences of length seq_len,
         so longer sequences are out-of-distribution and would produce garbage.
    """

    def __init__(self, video_tokenizer, dynamics_model, action_index_to_latents_fn,
                 video_index_to_latents_fn, context_length=4, num_maskgit_steps=12,
                 temperature=0.0, schedule_k=5.0):
        self.video_tokenizer = video_tokenizer
        self.dynamics_model = dynamics_model
        self.action_index_to_latents_fn = action_index_to_latents_fn
        self.video_index_to_latents_fn = video_index_to_latents_fn
        self.context_length = context_length
        self.num_maskgit_steps = num_maskgit_steps
        self.temperature = temperature
        self.schedule_k = schedule_k

        # Running buffers — initialized by .reset()
        self.latent_buffer = None   # [B, T_ctx, P, 5] — recent z_q frames
        self.action_buffer = None   # [B, T_ctx-1, 2] — recent action conditioning vectors
        self.device = None
        self.batch_size = None

    @torch.no_grad()
    def reset(self, initial_frames):
        """Seed the rollout buffer from a real starting clip.

        Args:
            initial_frames: [B, T, C, H, W] — a real clip (e.g. from ZeldaDataset), range [-1,1].
                            T must be >= context_length.
        """
        B, T = initial_frames.shape[:2]
        self.device = initial_frames.device
        self.batch_size = B

        assert T >= self.context_length, (
            f"Need at least {self.context_length} frames to seed context, got {T}"
        )

        # Encode all frames into z_q space
        z = self.video_tokenizer.encoder(initial_frames)    # [B, T, P, 5]
        z_q = self.video_tokenizer.quantizer(z)             # [B, T, P, 5]

        # Take the last context_length frames as our initial buffer
        self.latent_buffer = z_q[:, -self.context_length:].detach()  # [B, T_ctx, P, 5]

        # Initialize action buffer with null (zero) actions for the T_ctx-1 transitions
        # between the initial context frames. We don't know what real actions produced
        # these frames — zero is the same null-padding convention used in train_dynamics.py.
        action_dim = 2  # from config: action_dim=2
        self.action_buffer = torch.zeros(
            B, self.context_length - 1, action_dim, device=self.device
        )  # [B, T_ctx-1, 2]

    @torch.no_grad()
    def step(self, action_index):
        """Generate the next frame given an action.

        Args:
            action_index: integer in [0, 15] — which latent action to apply

        Returns:
            frame_pixels: [B, C, H, W] — generated frame in [-1, 1] range
            frame_np: [H, W, 3] numpy array in [0, 1] — first batch element, ready for display
        """
        assert self.latent_buffer is not None, "Call .reset() before .step()"

        B = self.batch_size
        T_ctx = self.context_length

        # 1. Convert action index to continuous conditioning vector
        action_cond = action_index_to_conditioning(
            action_index, self.action_index_to_latents_fn, B, self.device,
        )  # [B, 1, 2]

        # 2. Build the full conditioning tensor for dynamics_model.generate.
        # Convention from train_dynamics.py's prepare_batch:
        #   - Frame 0 gets a null (zero) action
        #   - Frame t gets action a_{t-1} (the action that transitions from frame t-1 to t)
        # So for T_ctx context frames + 1 horizon frame, we need T_ctx+1 conditioning vectors:
        #   [null, action_buffer[0], ..., action_buffer[T_ctx-2], new_action]
        null_action = torch.zeros(B, 1, 2, device=self.device)  # [B, 1, 2]
        conditioning = torch.cat([
            null_action,          # for context frame 0
            self.action_buffer,   # for context frames 1..T_ctx-1
            action_cond,          # for the new horizon frame
        ], dim=1)  # [B, T_ctx + 1, 2]

        # 3. Call dynamics_model.generate to predict the next frame's z_q
        full_latents = self.dynamics_model.generate(
            context_latents=self.latent_buffer,           # [B, T_ctx, P, 5]
            num_steps=self.num_maskgit_steps,
            index_to_latents_fn=self.video_index_to_latents_fn,
            conditioning=conditioning,                     # [B, T_ctx+1, 2]
            temperature=self.temperature,
            schedule_k=self.schedule_k,
            horizon=1,
        )  # [B, T_ctx+1, P, 5]

        # Extract the newly generated frame (last position)
        new_z_q = full_latents[:, -1:].detach()  # [B, 1, P, 5]

        # 4. Slide the window: drop oldest, append newest
        self.latent_buffer = torch.cat([
            self.latent_buffer[:, 1:],  # drop oldest frame
            new_z_q,                    # append new frame
        ], dim=1).detach()  # [B, T_ctx, P, 5] — same length as before

        # Update action buffer: drop oldest action, append the new one
        self.action_buffer = torch.cat([
            self.action_buffer[:, 1:],  # drop oldest action
            action_cond,                # append new action
        ], dim=1).detach()  # [B, T_ctx-1, 2] — same length as before

        # 5. Decode to pixels
        frame_pixels = self.video_tokenizer.decoder(new_z_q)  # [B, 1, C, H, W]
        frame_pixels = frame_pixels.squeeze(1)                 # [B, C, H, W]

        # Prepare numpy version for display: first batch element, [0,1] range, HWC
        frame_display = ((frame_pixels[0] + 1) / 2).clamp(0, 1)  # [C, H, W] in [0, 1]
        frame_np = frame_display.cpu().permute(1, 2, 0).numpy()   # [H, W, 3]

        return frame_pixels, frame_np
