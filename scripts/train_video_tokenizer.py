"""
Train the Video Tokenizer (FSQ-VAE with Space-Time Transformer backbone).

This is Phase 3 of the Mini Genie roadmap.

Usage:
    python scripts/train_video_tokenizer.py
    python scripts/train_video_tokenizer.py --config configs/video_tokenizer.yaml
    python scripts/train_video_tokenizer.py --dataset sonic --epochs 50

Key training details:
    - Reconstruction loss: MSE between input and reconstructed frames
    - Optimizer: AdamW with cosine annealing + linear warmup
    - Gradient clipping at max_norm=1.0
    - Saves reconstructions side-by-side every vis_every epochs
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from tokenizer.video_tokenizer import VideoTokenizer

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_reconstruction_grid(frames, recon, epoch, save_dir, n_show=4):
    """Save side-by-side comparison of input vs reconstructed frames."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frames = ((frames + 1) / 2).clamp(0, 1).cpu()  # [-1,1] → [0,1]
    recon = ((recon + 1) / 2).clamp(0, 1).cpu()

    n_show = min(n_show, frames.shape[0])
    T = frames.shape[1]

    fig, axes = plt.subplots(n_show * 2, T, figsize=(2 * T, 2 * n_show * 2))
    for i in range(n_show):
        for t in range(T):
            # Original
            ax_orig = axes[i * 2, t]
            ax_orig.imshow(frames[i, t].permute(1, 2, 0).numpy())
            ax_orig.axis("off")
            if i == 0:
                ax_orig.set_title(f"t={t}", fontsize=8)
            if t == 0:
                ax_orig.set_ylabel("input", fontsize=8)

            # Reconstruction
            ax_recon = axes[i * 2 + 1, t]
            ax_recon.imshow(recon[i, t].permute(1, 2, 0).numpy())
            ax_recon.axis("off")
            if t == 0:
                ax_recon.set_ylabel("recon", fontsize=8)

    plt.suptitle(f"Epoch {epoch} — Input (odd rows) vs Reconstruction (even rows)", fontsize=10)
    plt.tight_layout()
    path = save_dir / f"recon_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def compute_codebook_usage(model, frames):
    """Check how many unique FSQ tokens are used — low usage = codebook collapse."""
    with torch.no_grad():
        indices = model.tokenize(frames)
    unique = indices.unique()
    return len(unique), model.codebook_size


def main():
    parser = argparse.ArgumentParser(description="Train Video Tokenizer")
    parser.add_argument("--config", default="configs/video_tokenizer.yaml")
    parser.add_argument("--dataset", default=None, help="Override dataset from config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Load configs ────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)

    # CLI overrides
    if args.dataset:
        cfg.dataset = args.dataset
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr

    dataset_name = cfg.dataset
    print(f"Training Video Tokenizer on: {dataset_name}")
    print(f"Config: embed_dim={cfg.embed_dim}, num_blocks={cfg.num_blocks}, "
          f"latent_dim={cfg.latent_dim}, num_bins={cfg.num_bins}, "
          f"codebook_size={cfg.num_bins ** cfg.latent_dim}")

    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ─────────────────────────────────────────────────
    DatasetCls = DATASET_CLASSES[dataset_name]

    train_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name],
        seq_len=data_cfg.seq_len,
        frame_size=cfg.frame_size,
        frame_skip=data_cfg.frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="train",
        train_frac=data_cfg.train_frac,
    )
    val_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name],
        seq_len=data_cfg.seq_len,
        frame_size=cfg.frame_size,
        frame_skip=data_cfg.frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="val",
        train_frac=data_cfg.train_frac,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} sequences, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} sequences, {len(val_loader)} batches")

    # ── Model ───────────────────────────────────────────────────
    model = VideoTokenizer(
        frame_size=cfg.frame_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        latent_dim=cfg.latent_dim,
        num_bins=cfg.num_bins,
        max_frames=cfg.max_frames,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── Optimizer & Scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    start_epoch = 0

    # ── Resume from checkpoint ──────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # ── Checkpoint dir ──────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = ckpt_dir / "visualizations"

    # ── Training loop ───────────────────────────────────────────
    best_val_loss = float("inf")
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, frames in enumerate(pbar):
            frames = frames.to(device)                        # [B, T, C, H, W]

            loss, recon = model(frames)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

        avg_train_loss = epoch_loss / max(1, n_batches)

        # ── Validation ──────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        val_frames_for_vis = None

        with torch.no_grad():
            for frames in val_loader:
                frames = frames.to(device)
                loss, recon = model(frames)
                val_loss_total += loss.item()
                val_batches += 1
                if val_frames_for_vis is None:
                    val_frames_for_vis = frames[:4]

        avg_val_loss = val_loss_total / max(1, val_batches)

        # ── Codebook usage ──────────────────────────────────────
        if val_frames_for_vis is not None:
            n_used, codebook_sz = compute_codebook_usage(model, val_frames_for_vis)
            usage_pct = 100 * n_used / codebook_sz
        else:
            n_used, codebook_sz, usage_pct = 0, model.codebook_size, 0

        print(f"[Epoch {epoch+1:3d}] train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"codebook={n_used}/{codebook_sz} ({usage_pct:.1f}%)")

        # ── Save reconstruction visualization ───────────────────
        if (epoch + 1) % cfg.vis_every == 0 and val_frames_for_vis is not None:
            with torch.no_grad():
                _, vis_recon = model(val_frames_for_vis)
            path = save_reconstruction_grid(val_frames_for_vis, vis_recon, epoch + 1, vis_dir)
            print(f"  → Saved reconstruction comparison to {path}")

        # ── Save checkpoint ─────────────────────────────────────
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": OmegaConf.to_container(cfg),
            }, ckpt_path)
            print(f"  → Saved checkpoint to {ckpt_path}")

        # ── Track best ──────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_loss": avg_val_loss,
                "config": OmegaConf.to_container(cfg),
            }, best_path)
            print(f"  ★ New best val_loss={avg_val_loss:.4f}, saved to {best_path}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
