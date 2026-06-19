"""
Extended inspection of LAM action codes.

Runs on a larger sample (500 clips), saves more examples for dominant codes,
and computes a quantitative motion magnitude metric (mean absolute pixel difference)
for each transition.
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from omegaconf import OmegaConf

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from datasets.loaders.pole_position_dataset import PolePositionDataset
from inference.model_loader import load_pretrained_models

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset, "pole_position": PolePositionDataset}


def main():
    parser = argparse.ArgumentParser(description="Extended LAM action inspection")
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of clips to sample from validation set")
    parser.add_argument("--examples_per_action", type=int, default=30,
                        help="How many example pairs to save per action index")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    video_tokenizer, lam, _, _, _ = load_pretrained_models(args.config, device)

    # Load validation dataset
    dataset_name = cfg.dataset
    DatasetCls = DATASET_CLASSES[dataset_name]
    val_ds = DatasetCls(
        h5_path=data_cfg.paths[dataset_name],
        seq_len=data_cfg.seq_len,
        frame_size=vt_cfg.frame_size,
        frame_skip=data_cfg.frame_skip,
        load_start_index=data_cfg.load_start_index[dataset_name],
        split="val",
        train_frac=data_cfg.train_frac,
    )

    output_dir = Path("checkpoints/action_inspection_extended")
    output_dir.mkdir(parents=True, exist_ok=True)

    # action_examples[action_idx] = [(frame_t, frame_t+1, diff), ...]
    action_examples = defaultdict(list)
    action_counts = defaultdict(int)
    action_diff_sums = defaultdict(float)

    print(f"Inspecting {args.num_samples} clips...")
    num_clips = min(args.num_samples, len(val_ds))

    with torch.no_grad():
        for i in range(num_clips):
            clip = val_ds[i].unsqueeze(0).to(device)  # [1, T, C, H, W]

            z = video_tokenizer.encoder(clip)            # [1, T, P, 5]
            z_q = video_tokenizer.quantizer(z)           # [1, T, P, 5]
            z_embed = video_tokenizer.decoder.latent_embed(z_q)  # [1, T, P, 32]

            action_indices = lam.infer_actions(z_embed)  # [1, T-1]

            T = clip.shape[1]
            frames_np = ((clip[0] + 1) / 2).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

            for t in range(T - 1):
                action_idx = action_indices[0, t].item()
                ft = frames_np[t]
                ft1 = frames_np[t + 1]
                
                # Compute mean absolute pixel difference in [0, 1] scale
                diff = np.mean(np.abs(ft1 - ft))
                
                action_counts[action_idx] += 1
                action_diff_sums[action_idx] += diff
                
                if len(action_examples[action_idx]) < args.examples_per_action:
                    action_examples[action_idx].append((ft, ft1, diff, i, t))

    print("\n" + "=" * 50)
    print("EXTENDED ACTION INDEX DISTRIBUTION")
    print("=" * 50)
    total = sum(action_counts.values())
    
    # Store stats to print neatly
    stats = []
    for idx in sorted(action_counts.keys()):
        count = action_counts[idx]
        pct = 100.0 * count / total
        mean_diff = action_diff_sums[idx] / count
        stats.append((idx, count, pct, mean_diff))
        
    for idx, count, pct, mean_diff in stats:
        bar = "█" * int(pct / 2)
        print(f"  Action {idx:2d}: {count:4d} ({pct:5.1f}%) | Mean Diff: {mean_diff:.4f} {bar}")

    unused = set(range(16)) - set(action_counts.keys())
    if unused:
        print(f"\n  Unused action indices: {sorted(unused)}")

    # Try to save images
    try:
        import matplotlib.pyplot as plt

        print(f"\nSaving example images to {output_dir}/...")
        for action_idx in sorted(action_examples.keys()):
            examples = action_examples[action_idx]
            n = len(examples)
            
            # Create a 2x(n//2) grid of subplots for easier viewing of many examples
            cols = 5
            rows = int(np.ceil(n / cols))
            
            fig, axes = plt.subplots(rows, cols * 2, figsize=(3 * cols * 2, 3 * rows))
            # Flatten axes for easy iteration, handle cases where there's only 1 row
            if rows == 1 and cols * 2 == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for ax in axes:
                ax.axis('off')

            for idx, (ft, ft1, diff, clip_i, t) in enumerate(examples):
                ax_t = axes[idx * 2]
                ax_t1 = axes[idx * 2 + 1]
                
                ax_t.imshow(ft)
                ax_t.set_title(f"t={t} (clip {clip_i})", fontsize=8)
                
                ax_t1.imshow(ft1)
                ax_t1.set_title(f"diff={diff:.4f}", fontsize=8)

            count = action_counts[action_idx]
            pct = 100.0 * count / total
            mean_diff = action_diff_sums[action_idx] / count
            fig.suptitle(f"Action {action_idx} — {count} occurrences ({pct:.1f}%) | Mean Diff: {mean_diff:.4f}", fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / f"action_{action_idx:02d}.png", dpi=100)
            plt.close()

        print(f"Done. Inspect images in {output_dir}/")

    except ImportError:
        print("\nmatplotlib not installed — cannot save images.")


if __name__ == "__main__":
    main()
