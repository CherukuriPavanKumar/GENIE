"""
Inspect LAM action codes — visualize what each action index actually does.

The LAM discovered 16 action codes from unlabeled gameplay. This script loads
real consecutive frame pairs from the validation set, runs lam.infer_actions
to get the action index assigned to each transition, and saves side-by-side
images organized by action index so you can manually see which index corresponds
to which movement direction.

Usage:
    python tests/inspect_actions.py
    python tests/inspect_actions.py --config configs/rollout.yaml --num_samples 20

After inspecting the output images in checkpoints/action_inspection/, update
the ACTION_KEY_MAP in scripts/play.py with the correct mappings.
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
from inference.model_loader import load_pretrained_models

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset}


def main():
    parser = argparse.ArgumentParser(description="Inspect LAM action codes")
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of clips to sample from validation set")
    parser.add_argument("--examples_per_action", type=int, default=3,
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

    output_dir = Path("checkpoints/action_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect action assignments
    # action_examples[action_idx] = [(frame_t, frame_t+1), ...]
    action_examples = defaultdict(list)
    action_counts = defaultdict(int)

    print(f"Inspecting {args.num_samples} clips...")
    num_clips = min(args.num_samples, len(val_ds))

    with torch.no_grad():
        for i in range(num_clips):
            clip = val_ds[i].unsqueeze(0).to(device)  # [1, T, C, H, W]

            # Get z_q and z_embed for LAM
            z = video_tokenizer.encoder(clip)            # [1, T, P, 5]
            z_q = video_tokenizer.quantizer(z)           # [1, T, P, 5]
            z_embed = video_tokenizer.decoder.latent_embed(z_q)  # [1, T, P, 32]

            # Infer action indices
            action_indices = lam.infer_actions(z_embed)  # [1, T-1]

            # Store frame pairs with their action assignments
            T = clip.shape[1]
            frames_np = ((clip[0] + 1) / 2).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

            for t in range(T - 1):
                action_idx = action_indices[0, t].item()
                action_counts[action_idx] += 1
                if len(action_examples[action_idx]) < args.examples_per_action:
                    action_examples[action_idx].append((
                        frames_np[t],      # [H, W, 3]
                        frames_np[t + 1],  # [H, W, 3]
                        i, t,
                    ))

    # Print action distribution
    print("\n" + "=" * 50)
    print("ACTION INDEX DISTRIBUTION")
    print("=" * 50)
    total = sum(action_counts.values())
    for idx in sorted(action_counts.keys()):
        count = action_counts[idx]
        pct = 100.0 * count / total
        bar = "█" * int(pct / 2)
        print(f"  Action {idx:2d}: {count:4d} ({pct:5.1f}%) {bar}")

    unused = set(range(16)) - set(action_counts.keys())
    if unused:
        print(f"\n  Unused action indices: {sorted(unused)}")

    # Try to save images (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        print(f"\nSaving example images to {output_dir}/...")
        for action_idx in sorted(action_examples.keys()):
            examples = action_examples[action_idx]
            n = len(examples)

            fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
            if n == 1:
                axes = axes.reshape(1, 2)

            for row, (ft, ft1, clip_i, t) in enumerate(examples):
                axes[row, 0].imshow(ft)
                axes[row, 0].set_title(f"frame t (clip {clip_i}, t={t})", fontsize=8)
                axes[row, 0].axis("off")

                axes[row, 1].imshow(ft1)
                axes[row, 1].set_title(f"frame t+1", fontsize=8)
                axes[row, 1].axis("off")

            count = action_counts[action_idx]
            pct = 100.0 * count / total
            fig.suptitle(f"Action {action_idx} — {count} occurrences ({pct:.1f}%)", fontsize=10)
            plt.tight_layout()
            plt.savefig(output_dir / f"action_{action_idx:02d}.png", dpi=150)
            plt.close()

        print(f"Done. Inspect images in {output_dir}/ and update ACTION_KEY_MAP in scripts/play.py")

    except ImportError:
        print("\nmatplotlib not installed — cannot save images. Install with: pip install matplotlib")
        print("Action distribution printed above is still useful for analysis.")


if __name__ == "__main__":
    main()
