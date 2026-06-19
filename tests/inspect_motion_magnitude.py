"""
Direct motion magnitude diagnostic.

Computes the mean absolute pixel difference between consecutive frames in the
validation dataset, independent of the models, to diagnose if the data itself
is mostly static at the current frame_skip.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Dataset motion magnitude")
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of clips to evaluate")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)

    dataset_name = cfg.dataset
    DatasetCls = DATASET_CLASSES[dataset_name]
    
    # We load with the current frame_skip configured in data_cfg
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

    diffs = []

    for i in range(num_clips):
        clip = val_ds[i].numpy()  # [T, C, H, W]
        # Convert [-1, 1] to [0, 1] range for intuitive diffs
        clip = ((clip + 1) / 2).clip(0, 1)
        
        T = clip.shape[0]
        for t in range(T - 1):
            ft = clip[t]
            ft1 = clip[t + 1]
            diff = np.mean(np.abs(ft1 - ft))
            diffs.append(diff)
            
    diffs = np.array(diffs)
    
    print("\n" + "=" * 50)
    print(f"MOTION MAGNITUDE STATS (frame_skip={frame_skip})")
    print("=" * 50)
    print(f"Total frame pairs: {len(diffs)}")
    print(f"Mean pixel diff:   {diffs.mean():.4f}")
    print(f"Median pixel diff: {np.median(diffs):.4f}")
    print(f"Max pixel diff:    {diffs.max():.4f}")
    
    # Define arbitrary thresholds for "static" vs "moving"
    # A mean absolute difference of 0.01 is about a 2.5/255 pixel difference across the whole frame.
    threshold_static = 0.005
    threshold_low_motion = 0.015
    
    static_count = np.sum(diffs < threshold_static)
    low_motion_count = np.sum((diffs >= threshold_static) & (diffs < threshold_low_motion))
    high_motion_count = np.sum(diffs >= threshold_low_motion)
    
    n = len(diffs)
    print("\nDistribution of Motion:")
    print(f"  < 0.005 (Near Static):   {static_count:5d} ({100 * static_count / n:5.1f}%)")
    print(f"  0.005 - 0.015 (Low):     {low_motion_count:5d} ({100 * low_motion_count / n:5.1f}%)")
    print(f"  > 0.015 (Meaningful):    {high_motion_count:5d} ({100 * high_motion_count / n:5.1f}%)")

    try:
        import matplotlib.pyplot as plt
        output_dir = Path("checkpoints/action_inspection_extended")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.hist(diffs, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of Pixel Differences between Consecutive Frames (frame_skip={frame_skip})")
        plt.xlabel("Mean Absolute Pixel Difference")
        plt.ylabel("Count")
        plt.axvline(x=threshold_static, color='red', linestyle='--', label=f'Static Threshold ({threshold_static})')
        plt.axvline(x=threshold_low_motion, color='orange', linestyle='--', label=f'Low Motion Threshold ({threshold_low_motion})')
        plt.legend()
        
        out_path = output_dir / f"motion_magnitude_hist_skip{frame_skip}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nSaved histogram to {out_path}")
        
    except ImportError:
        print("\nmatplotlib not installed — cannot save histogram.")


if __name__ == "__main__":
    main()
