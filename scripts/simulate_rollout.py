"""
Headless Rollout Simulator.

Since cloud environments (like Lightning AI Studio) do not have a physical screen 
to open a PyGame window, this script runs the interactive rollout engine "blindly"
by feeding it a predefined sequence of actions and saving the result as a GIF.

Usage:
    python scripts/simulate_rollout.py --actions 9 9 9 14 14 14
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from datasets.loaders.pole_position_dataset import PolePositionDataset
from inference.model_loader import load_pretrained_models
from inference.rollout import RolloutState

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset, "pole_position": PolePositionDataset}


def main():
    parser = argparse.ArgumentParser(description="Simulate Rollout (Headless)")
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--actions", nargs='+', type=int, default=[9, 9, 9, 14, 14, 14],
                        help="List of action indices (0-15) to execute in sequence")
    parser.add_argument("--output", default="rollout.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=int, default=10, help="Framerate for the output GIF")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    video_tokenizer, lam, dynamics_model, video_idx_to_lat, action_idx_to_lat = \
        load_pretrained_models(args.config, device)

    # Initialize rollout state
    rollout = RolloutState(
        video_tokenizer=video_tokenizer,
        dynamics_model=dynamics_model,
        action_index_to_latents_fn=action_idx_to_lat,
        video_index_to_latents_fn=video_idx_to_lat,
        context_length=cfg.context_length,
        num_maskgit_steps=cfg.num_maskgit_steps,
        temperature=cfg.temperature,
        schedule_k=cfg.schedule_k,
    )

    # Seed from real data
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
    
    starting_clip = val_ds[0].unsqueeze(0).to(device)
    rollout.reset(starting_clip)

    frames_to_save = []

    # Decode initial context frames
    with torch.no_grad():
        initial_pixels = video_tokenizer.decoder(rollout.latent_buffer)  # [1, T_ctx, C, H, W]
    
    for t in range(initial_pixels.shape[1]):
        frame_np = ((initial_pixels[0, t] + 1) / 2).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        frames_to_save.append(frame_np)

    print(f"Loaded {len(frames_to_save)} initial context frames.")
    print(f"Executing action sequence: {args.actions}")

    # Step through actions
    for i, action in enumerate(args.actions):
        print(f"Step {i+1}/{len(args.actions)} — Action {action}...")
        _, frame_np = rollout.step(action)
        frames_to_save.append(frame_np)

    # Save to GIF
    print(f"Saving {len(frames_to_save)} frames to {args.output}...")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    
    # Render first frame
    im = ax.imshow(frames_to_save[0])

    def update(frame_idx):
        im.set_array(frames_to_save[frame_idx])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames_to_save), interval=1000//args.fps, blit=True)
    ani.save(args.output, writer="pillow", fps=args.fps)
    plt.close()
    
    print(f"✅ Success! Open {args.output} to view your rollout.")

if __name__ == "__main__":
    main()
