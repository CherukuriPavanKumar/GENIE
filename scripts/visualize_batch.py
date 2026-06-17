"""
Phase 2 validation gate. Confirms the data pipeline produces correct,
visually sane sequences before any model touches it.

Usage:
    python scripts/visualize_batch.py --dataset zelda
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.loaders.sonic_dataset import SonicDataset
from datasets.loaders.zelda_dataset import ZeldaDataset

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_CLASSES.keys(), default="zelda")
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/data.yaml")
    DatasetCls = DATASET_CLASSES[args.dataset]

    ds = DatasetCls(
        h5_path=cfg.paths[args.dataset],
        seq_len=cfg.seq_len,
        frame_size=cfg.frame_size,
        frame_skip=cfg.frame_skip,
        load_start_index=cfg.load_start_index[args.dataset],
        split="train",
        train_frac=cfg.train_frac,
    )
    print(f"Dataset length: {len(ds)} sequences")
    print(f"Native frame shape in h5: {ds.native_shape}")

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    frames = next(iter(loader))  # [B, T, C, H, W], range [-1, 1]

    print(f"Batch shape: {tuple(frames.shape)}")
    print(f"Value range: [{frames.min():.3f}, {frames.max():.3f}]")
    assert frames.shape[1] == cfg.seq_len, "seq_len mismatch -- check frame_skip math"
    assert frames.min() >= -1.001 and frames.max() <= 1.001, "normalization is off"

    display = ((frames + 1) / 2).clamp(0, 1)  # back to [0, 1] for plotting
    n_show = min(4, display.shape[0])

    fig, axes = plt.subplots(n_show, cfg.seq_len, figsize=(2 * cfg.seq_len, 2 * n_show))
    for i in range(n_show):
        for t in range(cfg.seq_len):
            ax = axes[i, t] if n_show > 1 else axes[t]
            ax.imshow(display[i, t].permute(1, 2, 0).numpy())
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={t}")
    plt.suptitle(f"{args.dataset} -- {n_show} sequences x {cfg.seq_len} frames")
    plt.tight_layout()

    out_path = f"batch_check_{args.dataset}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    print(
        "Check: consecutive frames (left to right per row) should show smooth, "
        "small game-state changes -- not random unrelated frames, not near-"
        "identical frozen frames. If frames look frozen, frame_skip is too low "
        "for this fps; if they look like unrelated jump-cuts, frame_skip is too high."
    )


if __name__ == "__main__":
    main()
