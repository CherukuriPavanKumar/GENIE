"""
Inspect Tokenizer Motion Preservation.

Directly tests whether the Video Tokenizer's FSQ latent bottleneck preserves the
magnitude of motion present in the raw pixel data, or if it flattens it.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import scipy.stats
from omegaconf import OmegaConf

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from inference.model_loader import load_pretrained_models

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Tokenizer Motion Preservation")
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of clips to evaluate")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen pretrained models
    video_tokenizer, _, _, _, _ = load_pretrained_models(args.config, device)

    dataset_name = cfg.dataset
    DatasetCls = DATASET_CLASSES[dataset_name]
    frame_skip = data_cfg.frame_skip
    
    val_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name],
        seq_len=data_cfg.seq_len,
        frame_size=vt_cfg.frame_size,
        frame_skip=frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="val",
        train_frac=data_cfg.train_frac,
    )

    print(f"Evaluating {args.num_samples} clips from {dataset_name} validation set (frame_skip={frame_skip})...")
    num_clips = min(args.num_samples, len(val_ds))

    # To store results
    pixel_diffs = []
    latent_diffs = []
    
    # To store metadata for top-n visualization
    pair_metadata = []

    with torch.no_grad():
        for i in range(num_clips):
            clip = val_ds[i].unsqueeze(0).to(device)  # [1, T, C, H, W]
            
            # Compute latent space representation for all frames in clip at once
            z = video_tokenizer.encoder(clip)        # [1, T, P, 5]
            z_q = video_tokenizer.quantizer(z)       # [1, T, P, 5]
            
            # Move to CPU numpy for diff computations
            clip_np = ((clip[0] + 1) / 2).clamp(0, 1).cpu().numpy()  # [T, C, H, W] in [0, 1]
            z_q_np = z_q[0].cpu().numpy()                            # [T, P, 5]
            
            T = clip_np.shape[0]
            for t in range(T - 1):
                ft_pix = clip_np[t]
                ft1_pix = clip_np[t + 1]
                p_diff = np.mean(np.abs(ft1_pix - ft_pix))
                
                ft_lat = z_q_np[t]
                ft1_lat = z_q_np[t + 1]
                l_diff = np.mean(np.abs(ft1_lat - ft_lat))
                
                pixel_diffs.append(p_diff)
                latent_diffs.append(l_diff)
                
                pair_metadata.append({
                    "clip_idx": i,
                    "t": t,
                    "p_diff": p_diff,
                    "l_diff": l_diff,
                    "ft_pix": ft_pix,     # Keeping in memory to save later, since num_clips is smallish
                    "ft1_pix": ft1_pix,
                })

    pixel_diffs = np.array(pixel_diffs)
    latent_diffs = np.array(latent_diffs)
    n_pairs = len(pixel_diffs)

    print("\n" + "=" * 60)
    print("TOKENIZER MOTION PRESERVATION ANALYSIS")
    print("=" * 60)
    
    # ── Correlation ───────────────────────────────────────────────────────────
    correlation, p_value = scipy.stats.pearsonr(pixel_diffs, latent_diffs)
    print(f"Total frame pairs analyzed: {n_pairs}")
    print(f"Pearson Correlation (Pixel vs Latent Diff): {correlation:.4f} (p={p_value:.4e})")
    
    # ── Top 5% vs Bottom 5% ───────────────────────────────────────────────────
    # Sort pairs by pixel difference
    sorted_indices = np.argsort(pixel_diffs)
    top_5_percent_idx = sorted_indices[int(n_pairs * 0.95):]
    bottom_5_percent_idx = sorted_indices[:int(n_pairs * 0.05)]
    
    top_pixel = pixel_diffs[top_5_percent_idx]
    top_latent = latent_diffs[top_5_percent_idx]
    
    bot_pixel = pixel_diffs[bottom_5_percent_idx]
    bot_latent = latent_diffs[bottom_5_percent_idx]
    
    print("\n--- Bottom 5% Pixel Motion (Near-Static) ---")
    print(f"  Pixel  Diff -> mean: {bot_pixel.mean():.4f}, min: {bot_pixel.min():.4f}, max: {bot_pixel.max():.4f}")
    print(f"  Latent Diff -> mean: {bot_latent.mean():.4f}, median: {np.median(bot_latent):.4f}, min: {bot_latent.min():.4f}, max: {bot_latent.max():.4f}")

    print("\n--- Top 5% Pixel Motion (Highest Motion) ---")
    print(f"  Pixel  Diff -> mean: {top_pixel.mean():.4f}, min: {top_pixel.min():.4f}, max: {top_pixel.max():.4f}")
    print(f"  Latent Diff -> mean: {top_latent.mean():.4f}, median: {np.median(top_latent):.4f}, min: {top_latent.min():.4f}, max: {top_latent.max():.4f}")

    # Calculate magnitude multiplier
    if bot_latent.mean() > 0:
        latent_multiplier = top_latent.mean() / bot_latent.mean()
    else:
        latent_multiplier = float('inf')
        
    pixel_multiplier = top_pixel.mean() / max(bot_pixel.mean(), 1e-8)

    print(f"\nMotion Multiplier (Top 5% vs Bottom 5%):")
    print(f"  Pixel space motion multiplied by:  {pixel_multiplier:.1f}x")
    print(f"  Latent space motion multiplied by: {latent_multiplier:.1f}x")

    # ── Visualizations ────────────────────────────────────────────────────────
    output_dir = Path("checkpoints/action_inspection_extended")
    hm_dir = output_dir / "highest_motion_pairs"
    hm_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        
        # 1. Scatter Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(pixel_diffs, latent_diffs, alpha=0.3, s=10, c='blue')
        plt.title(f"Pixel Motion vs Latent Motion\nPearson r = {correlation:.4f}")
        plt.xlabel("Mean Absolute Pixel Difference")
        plt.ylabel("Mean Absolute FSQ Latent Difference")
        plt.grid(True, alpha=0.3)
        scatter_path = output_dir / "pixel_vs_latent_diff_scatter.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"\nSaved scatter plot to {scatter_path}")
        
        # 2. Top 10 High Motion Heatmaps
        print(f"Saving top 10 highest motion pair visualizations to {hm_dir}/...")
        top_10_indices = sorted_indices[-10:][::-1]  # descending
        
        for rank, idx in enumerate(top_10_indices):
            meta = pair_metadata[idx]
            ft = meta["ft_pix"].transpose(1, 2, 0)   # [H, W, C]
            ft1 = meta["ft1_pix"].transpose(1, 2, 0) # [H, W, C]
            
            # Compute channel-mean absolute difference for heatmap
            diff_heatmap = np.mean(np.abs(ft1 - ft), axis=-1)  # [H, W]
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(ft)
            axes[0].set_title(f"Frame t")
            axes[0].axis('off')
            
            axes[1].imshow(ft1)
            axes[1].set_title(f"Frame t+1")
            axes[1].axis('off')
            
            # Use 'hot' colormap for heatmap
            im = axes[2].imshow(diff_heatmap, cmap='hot', vmin=0.0, vmax=0.2)
            axes[2].set_title(f"|t - t+1| (Mean Diff: {meta['p_diff']:.4f})")
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            fig.suptitle(f"Rank {rank+1} Highest Pixel Motion (Clip {meta['clip_idx']}, transition {meta['t']})")
            plt.tight_layout()
            plt.savefig(hm_dir / f"rank_{rank+1:02d}_clip{meta['clip_idx']}_t{meta['t']}.png", dpi=100)
            plt.close()
            
    except ImportError:
        print("\nmatplotlib not installed — cannot save plots.")

    # ── Final Conclusion ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL CONCLUSION")
    print("=" * 60)
    
    # We define arbitrary thresholds to output a clear, automated conclusion.
    # If correlation is > 0.5 and top-5% latent diff is > 1.5x the bottom-5%, the tokenizer is preserving motion.
    if correlation > 0.5 and latent_multiplier > 1.5:
        print("CONCLUSION: The evidence SUPPORTS HYPOTHESIS 1 (frame_skip fix would likely help).")
        print("Reasoning: The Pearson correlation is strong (> 0.5) and the Latent space explicitly ")
        print("differentiates between high-pixel-motion and low-pixel-motion pairs ")
        print(f"(top 5% latent diff is {latent_multiplier:.1f}x higher than bottom 5%). ")
        print("The Tokenizer IS preserving motion magnitude. The LAM's failure to learn is likely ")
        print("because the overall raw motion at frame_skip=4 is too small or unbalanced, not because ")
        print("the tokenizer flattens it. Retraining with a higher frame_skip is the recommended next step.")
    else:
        print("CONCLUSION: The evidence SUPPORTS HYPOTHESIS 2 (tokenizer is the bottleneck, frame_skip fix would NOT help).")
        print(f"Reasoning: The Pearson correlation is weak ({correlation:.4f}) and/or the Latent space ")
        print("fails to meaningfully differentiate high-pixel-motion from low-pixel-motion pairs ")
        print(f"(top 5% latent diff is only {latent_multiplier:.1f}x higher than bottom 5%, compared to ")
        print(f"a {pixel_multiplier:.1f}x difference in pixel space).")
        print("The Tokenizer collapses/flattens motion regardless of how much exists in the raw pixels. ")
        print("Increasing frame_skip will not solve this because the tokenizer will still compress it away. ")
        print("The real fix requires changing the Video Tokenizer's architecture or FSQ parameters.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
