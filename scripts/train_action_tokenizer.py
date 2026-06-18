"""
Train the Action Tokenizer (Latent Action Model).

This is Phase 4 of the Mini Genie roadmap.
It relies on a trained Video Tokenizer.

Usage:
    python scripts/train_action_tokenizer.py
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
from lam.latent_action_model import LatentActionModel

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_reconstruction_grid(frames, recon, actions, epoch, save_dir, n_show=4):
    """Save side-by-side comparison of input vs reconstructed frames."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frames = ((frames + 1) / 2).clamp(0, 1).cpu()  # [-1,1] → [0,1]
    recon = ((recon + 1) / 2).clamp(0, 1).cpu()
    actions = actions.cpu().numpy()

    n_show = min(n_show, frames.shape[0])
    T = frames.shape[1]

    fig, axes = plt.subplots(n_show * 2, T, figsize=(2 * T, 2 * n_show * 2))
    for i in range(n_show):
        action_seq = actions[i]
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
                ax_recon.set_ylabel(f"recon\nactions:{list(action_seq)}", fontsize=8)

    plt.suptitle(f"LAM Epoch {epoch} — Input (odd) vs Recon (even)", fontsize=10)
    plt.tight_layout()
    path = save_dir / f"lam_recon_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def compute_action_usage(actions, vocab_size):
    """Check how many unique action tokens are used."""
    unique = actions.unique()
    return len(unique), vocab_size


def main():
    parser = argparse.ArgumentParser(description="Train Action Tokenizer (LAM)")
    parser.add_argument("--config", default="configs/action_tokenizer.yaml")
    parser.add_argument("--dataset", default=None, help="Override dataset from config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Load configs ────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)

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
    print(f"Training LAM on: {dataset_name}")

    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ─────────────────────────────────────────────────
    DatasetCls = DATASET_CLASSES[dataset_name]

    train_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name],
        seq_len=data_cfg.seq_len,
        frame_size=vt_cfg.frame_size,
        frame_skip=data_cfg.frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="train",
        train_frac=data_cfg.train_frac,
    )
    val_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name],
        seq_len=data_cfg.seq_len,
        frame_size=vt_cfg.frame_size,
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

    # ── Video Tokenizer ─────────────────────────────────────────
    # We load the Video Tokenizer to embed the frames and decode the LAM predictions.
    print(f"Loading Pretrained Video Tokenizer from: {cfg.video_tokenizer_ckpt}")
    video_tokenizer = VideoTokenizer(
        frame_size=vt_cfg.frame_size, patch_size=vt_cfg.patch_size, embed_dim=vt_cfg.embed_dim, 
        num_heads=vt_cfg.num_heads, hidden_dim=vt_cfg.hidden_dim, num_blocks=vt_cfg.num_blocks, 
        latent_dim=vt_cfg.latent_dim, num_bins=vt_cfg.num_bins, max_frames=vt_cfg.max_frames,
    ).to(device)

    try:
        vt_ckpt = torch.load(cfg.video_tokenizer_ckpt, map_location=device, weights_only=False)
        video_tokenizer.load_state_dict(vt_ckpt["model"])
        print("Successfully loaded Video Tokenizer weights.")
    except Exception as e:
        print(f"WARNING: Could not load Video Tokenizer weights. ({e})")
        print("Ensure you have trained the Video Tokenizer first. Continuing with random weights for debugging.")

    # Freeze Video Tokenizer
    video_tokenizer.eval()
    for param in video_tokenizer.parameters():
        param.requires_grad = False

    # ── LAM Model ───────────────────────────────────────────────
    model = LatentActionModel(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        action_dim=cfg.action_dim,
        num_bins=cfg.num_bins,
        var_weight=cfg.var_weight,
        var_target=cfg.var_target,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LAM Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

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

            # 1. Embed frames using frozen Video Tokenizer Encoder
            with torch.no_grad():
                # We need the patch embeddings BEFORE FSQ for LAM training
                # video_tokenizer.encoder returns [B, T, P, latent_dim]
                # Wait, LAM needs embed_dim (the continuous embeddings, not the quantised ones ideally, 
                # or it works on the latent_dim. The roadmap says "Start with the pretrained Video Tokenizer encoder to get patch embeddings".
                # Actually, video_tokenizer.encoder returns [B, T, P, latent_dim]. 
                # Let's use these latent_dim vectors as the frame representation.
                # But wait, LAM config has embed_dim=32 matching VT embed_dim. 
                # If we use VT encoder output it's latent_dim (5). 
                # Let's adjust LAM to operate on the continuous embeddings just before the latent_head,
                # or we just pass it through the latent_embed from the Decoder to get back to embed_dim.
                # Let's just use encoder output [B, T, P, latent_dim] and then pass it through decoder's latent_embed to get [B, T, P, E]
                
                z_latents = video_tokenizer.encoder(frames) # [B, T, P, latent_dim]
                z_q = video_tokenizer.quantizer(z_latents) # [B, T, P, latent_dim]
                # Map back to continuous embed_dim to be processed by LAM
                z_embed = video_tokenizer.decoder.latent_embed(z_q) # [B, T, P, E]

            # 2. Train LAM
            total_loss, recon_loss, var_loss, action_latents_q = model(z_embed)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", rec=f"{recon_loss.item():.4f}", var=f"{var_loss.item():.4f}", lr=f"{lr_now:.2e}")

        avg_train_loss = epoch_loss / max(1, n_batches)

        # ── Validation ──────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        val_frames_for_vis = None
        all_actions = []

        with torch.no_grad():
            for frames in val_loader:
                frames = frames.to(device)
                
                z_latents = video_tokenizer.encoder(frames)
                z_q = video_tokenizer.quantizer(z_latents)
                z_embed = video_tokenizer.decoder.latent_embed(z_q)
                
                total_loss, recon_loss, var_loss, action_latents_q = model(z_embed)
                
                val_loss_total += total_loss.item()
                val_batches += 1
                
                # Get action indices
                action_idx = model.quantizer.get_indices_from_latents(action_latents_q)
                all_actions.append(action_idx)
                
                if val_frames_for_vis is None:
                    val_frames_for_vis = frames[:4]
                    val_z_embed = z_embed[:4]
                    val_actions = action_idx[:4]

        avg_val_loss = val_loss_total / max(1, val_batches)
        
        all_actions = torch.cat(all_actions, dim=0)

        # ── Action diversity ────────────────────────────────────
        n_used, action_vocab_sz = compute_action_usage(all_actions, model.action_vocab_size)
        usage_pct = 100 * n_used / action_vocab_sz

        print(f"[Epoch {epoch+1:3d}] train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"actions_used={n_used}/{action_vocab_sz} ({usage_pct:.1f}%)")

        # ── Save reconstruction visualization ───────────────────
        if (epoch + 1) % cfg.vis_every == 0 and val_frames_for_vis is not None:
            with torch.no_grad():
                # Get predicted z_embed from LAM
                # action_latents_q for vis
                action_latents = model.encoder(val_z_embed)
                action_latents_q = model.quantizer(action_latents)
                z_pred = model.decoder(val_z_embed[:, :1], action_latents_q, T=val_z_embed.shape[1])
                
                # We need to map z_pred [B, T, P, E] back to pixels.
                # In Video Tokenizer Decoder: x = x + spatial_pe + temporal_pe -> STT -> frame_head
                # But z_pred is BEFORE spatial/temporal PE. We should pass it through the rest of the decoder.
                # Actually, z_pred represents `z_embed`, which is just after `latent_embed`.
                # So we must add PE and pass through STT.
                B, T, P, E = z_pred.shape
                x = z_pred + video_tokenizer.decoder.spatial_pe
                x = x + video_tokenizer.decoder.temporal_pe[:, :T]
                x = video_tokenizer.decoder.transformer(x)
                vis_recon = video_tokenizer.decoder.frame_head(x)

            path = save_reconstruction_grid(val_frames_for_vis, vis_recon, val_actions, epoch + 1, vis_dir)
            print(f"  → Saved LAM reconstruction comparison to {path}")

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
