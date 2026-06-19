"""
Train the Dynamics Model (MaskGIT-style masked token prediction).

This is Phase 5 of the Mini Genie roadmap.  
Requires trained Video Tokenizer and Action Tokenizer checkpoints.

Usage:
    python scripts/train_dynamics.py
    python scripts/train_dynamics.py --config configs/dynamics.yaml
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from tokenizer.video_tokenizer import VideoTokenizer
from tokenizer.fsq import FiniteScalarQuantizer
from lam.latent_action_model import LatentActionModel
from dynamics.dynamics_model import DynamicsModel

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_index_to_latents_fn(quantizer):
    """Build a function that converts token indices back to FSQ latent vectors."""
    num_bins = quantizer.num_bins
    latent_dim = quantizer.latent_dim

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


def save_generation_grid(frames, epoch, save_dir, title="Generated"):
    """Save a grid of generated frames."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frames = ((frames + 1) / 2).clamp(0, 1).cpu()
    n_show = min(4, frames.shape[0])
    T = frames.shape[1]

    fig, axes = plt.subplots(n_show, T, figsize=(2 * T, 2 * n_show))
    for i in range(n_show):
        for t in range(T):
            ax = axes[i, t] if n_show > 1 else axes[t]
            ax.imshow(frames[i, t].permute(1, 2, 0).numpy())
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={t}", fontsize=8)
    plt.suptitle(f"{title} — Epoch {epoch}", fontsize=10)
    plt.tight_layout()
    path = save_dir / f"dynamics_gen_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Train Dynamics Model")
    parser.add_argument("--config", default="configs/dynamics.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # ── Load configs ────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)
    at_cfg = OmegaConf.load(cfg.action_tokenizer_config)

    if args.dataset: cfg.dataset = args.dataset
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr

    dataset_name = cfg.dataset
    print(f"Training Dynamics Model on: {dataset_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ─────────────────────────────────────────────────
    DatasetCls = DATASET_CLASSES[dataset_name]
    train_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name], seq_len=data_cfg.seq_len,
        frame_size=vt_cfg.frame_size, frame_skip=data_cfg.frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="train", train_frac=data_cfg.train_frac,
    )
    val_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name], seq_len=data_cfg.seq_len,
        frame_size=vt_cfg.frame_size, frame_skip=data_cfg.frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="val", train_frac=data_cfg.train_frac,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=data_cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=data_cfg.num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} sequences, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} sequences, {len(val_loader)} batches")

    # ── Load Frozen Video Tokenizer ─────────────────────────────
    video_tokenizer = VideoTokenizer(
        frame_size=vt_cfg.frame_size, patch_size=vt_cfg.patch_size, embed_dim=vt_cfg.embed_dim,
        num_heads=vt_cfg.num_heads, hidden_dim=vt_cfg.hidden_dim, num_blocks=vt_cfg.num_blocks,
        latent_dim=vt_cfg.latent_dim, num_bins=vt_cfg.num_bins, max_frames=vt_cfg.max_frames,
    ).to(device)
    try:
        vt_ckpt = torch.load(cfg.video_tokenizer_ckpt, map_location=device, weights_only=False)
        video_tokenizer.load_state_dict(vt_ckpt["model"])
        print(f"Loaded Video Tokenizer from {cfg.video_tokenizer_ckpt}")
    except Exception as e:
        print(f"WARNING: Could not load Video Tokenizer ({e}). Using random weights.")
    video_tokenizer.eval()
    for p in video_tokenizer.parameters(): p.requires_grad = False

    # ── Load Frozen LAM ─────────────────────────────────────────
    lam = LatentActionModel(
        embed_dim=at_cfg.embed_dim, num_heads=at_cfg.num_heads, hidden_dim=at_cfg.hidden_dim,
        num_blocks=at_cfg.num_blocks, action_dim=at_cfg.action_dim, num_bins=at_cfg.num_bins,
        var_weight=at_cfg.var_weight, var_target=at_cfg.var_target,
    ).to(device)
    try:
        at_ckpt = torch.load(cfg.action_tokenizer_ckpt, map_location=device, weights_only=False)
        lam.load_state_dict(at_ckpt["model"])
        print(f"Loaded Action Tokenizer from {cfg.action_tokenizer_ckpt}")
    except Exception as e:
        print(f"WARNING: Could not load Action Tokenizer ({e}). Using random weights.")
    lam.eval()
    for p in lam.parameters(): p.requires_grad = False

    # Helper: convert token indices → FSQ latent vectors
    index_to_latents = build_index_to_latents_fn(video_tokenizer.quantizer)

    # ── Dynamics Model ──────────────────────────────────────────
    model = DynamicsModel(
        embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks, latent_dim=vt_cfg.latent_dim, num_bins=vt_cfg.num_bins,
        action_dim=at_cfg.action_dim, max_frames=cfg.max_frames,
        frame_size=vt_cfg.frame_size, patch_size=vt_cfg.patch_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dynamics Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"Codebook size: {model.codebook_size}")

    # ── Optimizer & Scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = ckpt_dir / "visualizations"

    # ── Precompute helper: frames → (latents, targets, action_conditioning) ──
    def prepare_batch(frames):
        """Convert raw frames to dynamics model inputs using frozen VT + LAM."""
        with torch.no_grad():
            # Video Tokenizer: frames → FSQ latent vectors + token indices
            z = video_tokenizer.encoder(frames)         # [B, T, P, latent_dim]
            z_q = video_tokenizer.quantizer(z)          # [B, T, P, latent_dim] quantized
            targets = video_tokenizer.quantizer.get_indices_from_latents(z_q)  # [B, T, P]

            # LAM: frames → action latent vectors for conditioning
            z_embed = video_tokenizer.decoder.latent_embed(z_q)  # [B, T, P, embed_dim]
            action_latents = lam.encoder(z_embed)        # [B, T-1, action_dim]
            action_latents_q = lam.quantizer(action_latents)  # [B, T-1, action_dim]

            # Pad actions to match T (null action for first frame)
            B = frames.shape[0]
            null_action = torch.zeros(B, 1, action_latents_q.shape[-1], device=frames.device)
            conditioning = torch.cat([null_action, action_latents_q], dim=1)  # [B, T, action_dim]

        return z_q, targets, conditioning

    # ── Training loop ───────────────────────────────────────────
    best_val_loss = float("inf")
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for frames in pbar:
            frames = frames.to(device)
            latents, targets, conditioning = prepare_batch(frames)

            _, mask, loss = model(
                latents, conditioning=conditioning, targets=targets,
                training=True, mask_ratio_min=cfg.mask_ratio_min, mask_ratio_max=cfg.mask_ratio_max,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Token accuracy on masked positions
            with torch.no_grad():
                logits, _, _ = model(latents, conditioning=conditioning, training=False)
                preds = logits.argmax(dim=-1)  # [B, T, P]
                if mask is not None:
                    correct = ((preds == targets) & mask).float().sum()
                    total_masked = mask.float().sum().clamp_min(1)
                    acc = (correct / total_masked).item()
                else:
                    acc = 0.0

            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", lr=f"{lr_now:.2e}")

        avg_train_loss = epoch_loss / max(1, n_batches)
        avg_train_acc = epoch_acc / max(1, n_batches)

        # ── Validation ──────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_acc_total = 0.0
        val_batches = 0
        val_context = None

        with torch.no_grad():
            for frames in val_loader:
                frames = frames.to(device)
                latents, targets, conditioning = prepare_batch(frames)

                logits, mask, loss = model(
                    latents, conditioning=conditioning, targets=targets,
                    training=True, mask_ratio_min=cfg.mask_ratio_min, mask_ratio_max=cfg.mask_ratio_max,
                )

                preds = logits.argmax(dim=-1)
                if mask is not None:
                    correct = ((preds == targets) & mask).float().sum()
                    total_masked = mask.float().sum().clamp_min(1)
                    acc = (correct / total_masked).item()
                else:
                    acc = 0.0

                val_loss_total += loss.item()
                val_acc_total += acc
                val_batches += 1

                if val_context is None:
                    val_context = (frames[:4], latents[:4], conditioning[:4])

        avg_val_loss = val_loss_total / max(1, val_batches)
        avg_val_acc = val_acc_total / max(1, val_batches)

        print(f"[Epoch {epoch+1:3d}] train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.3f}  "
              f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.3f}")

        # ── Autoregressive generation visualization ─────────────
        if (epoch + 1) % cfg.vis_every == 0 and val_context is not None:
            vis_frames, vis_latents, vis_cond = val_context
            with torch.no_grad():
                # Use first 2 frames as context, predict remaining
                T_ctx = 2
                context = vis_latents[:, :T_ctx]
                cond = vis_cond[:, :T_ctx + 1]  # +1 for the predicted frame

                generated = model.generate(
                    context, num_steps=cfg.num_maskgit_steps,
                    index_to_latents_fn=lambda idx: index_to_latents(idx).to(device),
                    conditioning=cond, temperature=cfg.temperature,
                    schedule_k=cfg.schedule_k, horizon=1,
                )
                # Decode generated latents back to pixels
                gen_pixels = video_tokenizer.decoder(generated)

            path = save_generation_grid(gen_pixels, epoch + 1, vis_dir, "Context+Generated")
            print(f"  → Saved generation to {path}")

        # ── Checkpoint ──────────────────────────────────────────
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                "config": OmegaConf.to_container(cfg),
            }, ckpt_path)
            print(f"  → Saved checkpoint to {ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_loss": avg_val_loss, "config": OmegaConf.to_container(cfg),
            }, best_path)
            print(f"  ★ New best val_loss={avg_val_loss:.4f}, saved to {best_path}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
