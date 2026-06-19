"""
Model Loader — loads all three frozen pretrained models for inference.

Constructs VideoTokenizer, LatentActionModel, and DynamicsModel from config,
loads their trained checkpoints, freezes everything, and returns ready-to-use
models plus two index_to_latents helper functions.

Nothing in Phase 6 should ever be trained or accumulate gradients.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from tokenizer.video_tokenizer import VideoTokenizer
from lam.latent_action_model import LatentActionModel
from dynamics.dynamics_model import DynamicsModel


def build_index_to_latents_fn(num_bins, latent_dim):
    """Build a function that converts integer token indices back to FSQ latent vectors.

    This replicates the exact logic from scripts/train_dynamics.py's
    build_index_to_latents_fn, but takes raw num_bins/latent_dim instead of a
    quantizer object so it can be used for both the video tokenizer codebook
    (latent_dim=5, num_bins=4) and the action codebook (latent_dim=2, num_bins=4).

    The math:
        index -> decompose into base-num_bins digits -> bin_center = (digit + 0.5) / num_bins * 2 - 1
    This is the inverse of FiniteScalarQuantizer.get_indices_from_latents.
    """
    def index_to_latents(indices):
        # indices: [...] integer token IDs
        device = indices.device
        orig_shape = indices.shape
        flat = indices.reshape(-1)

        # Decompose base-num_bins number back to per-dimension bin indices
        bin_indices = []
        remaining = flat
        for d in range(latent_dim):
            bin_indices.append(remaining % num_bins)
            remaining = remaining // num_bins
        bin_indices = torch.stack(bin_indices, dim=-1).float()  # [N, latent_dim]

        # Convert bin indices back to latent values (bin centers in [-1, 1])
        latents = (bin_indices + 0.5) / num_bins * 2 - 1

        return latents.reshape(*orig_shape, latent_dim)

    return index_to_latents


def load_pretrained_models(config_path="configs/rollout.yaml", device=None, load_dynamics=True):
    """Load pretrained models for interactive rollout or inspection.

    Args:
        config_path: path to configs/rollout.yaml
        device: torch device (auto-detects cuda if available)
        load_dynamics: if False, skips loading the Dynamics Model and returns None

    Returns:
        video_tokenizer: frozen VideoTokenizer
        lam: frozen LatentActionModel
        dynamics_model: frozen DynamicsModel (or None)
        video_index_to_latents: callable(indices) -> latents for video codebook (latent_dim=5)
        action_index_to_latents: callable(indices) -> latents for action codebook (latent_dim=2)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(config_path)

    # Load component configs — same pattern as scripts/train_dynamics.py
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)
    at_cfg = OmegaConf.load(cfg.action_tokenizer_config)
    if load_dynamics:
        dyn_cfg = OmegaConf.load(cfg.dynamics_config)

    # ── Video Tokenizer ─────────────────────────────────────────
    vt_ckpt_path = Path(cfg.video_tokenizer_ckpt)
    if not vt_ckpt_path.exists():
        raise FileNotFoundError(
            f"Video Tokenizer checkpoint not found at: {vt_ckpt_path.resolve()}\n"
            f"Train it first with: python scripts/train_video_tokenizer.py"
        )

    video_tokenizer = VideoTokenizer(
        frame_size=vt_cfg.frame_size, patch_size=vt_cfg.patch_size, embed_dim=vt_cfg.embed_dim,
        num_heads=vt_cfg.num_heads, hidden_dim=vt_cfg.hidden_dim, num_blocks=vt_cfg.num_blocks,
        latent_dim=vt_cfg.latent_dim, num_bins=vt_cfg.num_bins, max_frames=vt_cfg.max_frames,
    ).to(device)
    vt_ckpt = torch.load(vt_ckpt_path, map_location=device, weights_only=False)
    video_tokenizer.load_state_dict(vt_ckpt["model"])
    video_tokenizer.eval()
    video_tokenizer.requires_grad_(False)
    print(f"✓ Loaded Video Tokenizer from {vt_ckpt_path}")

    # ── Latent Action Model ─────────────────────────────────────
    # Check both possible checkpoint paths
    at_ckpt_path = Path(cfg.action_tokenizer_ckpt)
    if not at_ckpt_path.exists():
        raise FileNotFoundError(
            f"Action Tokenizer (LAM) checkpoint not found at: {at_ckpt_path.resolve()}\n"
            f"Train it first with: python scripts/train_action_tokenizer.py"
        )

    lam = LatentActionModel(
        embed_dim=at_cfg.embed_dim, num_heads=at_cfg.num_heads, hidden_dim=at_cfg.hidden_dim,
        num_blocks=at_cfg.num_blocks, action_dim=at_cfg.action_dim, num_bins=at_cfg.num_bins,
        var_weight=at_cfg.var_weight, var_target=at_cfg.var_target,
    ).to(device)
    at_ckpt = torch.load(at_ckpt_path, map_location=device, weights_only=False)
    lam.load_state_dict(at_ckpt["model"])
    lam.eval()
    lam.requires_grad_(False)
    print(f"✓ Loaded LAM from {at_ckpt_path}")

    # ── Dynamics Model ──────────────────────────────────────────
    dynamics_model = None
    if load_dynamics:
        dyn_ckpt_path = Path(cfg.dynamics_ckpt)
        if not dyn_ckpt_path.exists():
            raise FileNotFoundError(
                f"Dynamics Model checkpoint not found at: {dyn_ckpt_path.resolve()}\n"
                f"Train it first with: python scripts/train_dynamics.py"
            )

        dynamics_model = DynamicsModel(
            embed_dim=dyn_cfg.embed_dim, num_heads=dyn_cfg.num_heads, hidden_dim=dyn_cfg.hidden_dim,
            num_blocks=dyn_cfg.num_blocks, latent_dim=vt_cfg.latent_dim, num_bins=vt_cfg.num_bins,
            action_dim=at_cfg.action_dim, max_frames=dyn_cfg.max_frames,
            frame_size=vt_cfg.frame_size, patch_size=vt_cfg.patch_size,
        ).to(device)
        dyn_ckpt = torch.load(dyn_ckpt_path, map_location=device, weights_only=False)
        dynamics_model.load_state_dict(dyn_ckpt["model"])
        dynamics_model.eval()
        dynamics_model.requires_grad_(False)
        print(f"✓ Loaded Dynamics Model from {dyn_ckpt_path}")

    # ── Index-to-latents helpers ────────────────────────────────
    # Video codebook: latent_dim=5, num_bins=4 → 1024 tokens
    video_index_to_latents = build_index_to_latents_fn(
        num_bins=vt_cfg.num_bins, latent_dim=vt_cfg.latent_dim,
    )
    # Action codebook: latent_dim=2, num_bins=4 → 16 actions
    action_index_to_latents = build_index_to_latents_fn(
        num_bins=at_cfg.num_bins, latent_dim=at_cfg.action_dim,
    )

    return video_tokenizer, lam, dynamics_model, video_index_to_latents, action_index_to_latents
