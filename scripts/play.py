"""
Interactive play demo — MiniGenie world model rollout with pygame.

Loads the three frozen pretrained models (Video Tokenizer, LAM, Dynamics Model),
seeds the context buffer from a real Zelda validation clip, and lets you press
keys to step the world forward one frame at a time.

IMPORTANT LIMITATIONS (read before running):
  - Generation is NOT real-time. Each frame requires num_maskgit_steps (default 12)
    full transformer forward passes through the Dynamics Model. Expect ~0.5-3
    seconds per frame on a T4 GPU, slower on CPU.
  - Reconstructed frames are 64x64 and inherently blurry — this is a limitation
    of the Phase 3 Video Tokenizer's reconstruction quality, not a bug.
  - The action-to-key mapping below is a PLACEHOLDER. The LAM discovered these
    16 action codes from unlabeled gameplay — there is no ground-truth mapping
    from "action index 7" to "press RIGHT". Run tests/inspect_actions.py first
    to visually inspect which action index corresponds to which movement direction,
    then update ACTION_KEY_MAP accordingly.

Usage:
    python scripts/play.py
    python scripts/play.py --config configs/rollout.yaml
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from datasets.loaders.pole_position_dataset import PolePositionDataset
from inference.model_loader import load_pretrained_models
from inference.rollout import RolloutState

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset, "pole_position": PolePositionDataset}

# ── ACTION KEY MAP ──────────────────────────────────────────────
# PLACEHOLDER MAPPING — these assignments are arbitrary because the LAM
# learned action codes without labels. Run tests/inspect_actions.py first
# to see what each action index actually does, then update this mapping.
#
# Format: pygame key constant -> action index (0-15)
# We import pygame key constants lazily inside main() to avoid import
# errors when pygame isn't installed (e.g. during testing).
ACTION_KEY_MAP_INDICES = {
    "K_w": 0,      # Up (guess)
    "K_a": 1,      # Left (guess)
    "K_s": 2,      # Down (guess)
    "K_d": 3,      # Right (guess)
    "K_SPACE": 4,   # Idle / no-op (guess)
}
DEFAULT_ACTION = 4  # Action used when no key is pressed (idle guess)


def get_starting_clip(cfg, vt_cfg, data_cfg, device):
    """Load a real starting clip from the dataset validation split."""
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
    # Grab the first clip and add batch dimension
    clip = val_ds[0].unsqueeze(0).to(device)  # [1, T, C, H, W]
    return clip, val_ds


def main():
    parser = argparse.ArgumentParser(description="MiniGenie Interactive Play")
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()

    # ── Import pygame ───────────────────────────────────────────
    try:
        import pygame
    except ImportError:
        print("ERROR: pygame is required for interactive play.")
        print("Install it with: pip install pygame")
        sys.exit(1)

    # ── Load config ─────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MiniGenie Interactive World Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"MaskGIT steps per frame: {cfg.num_maskgit_steps}")
    print(f"Temperature: {cfg.temperature}")
    print()
    print("⚠  Each frame requires multiple transformer forward passes.")
    print("   Expect ~0.5-3 seconds per frame on GPU, much slower on CPU.")
    print()

    # ── Load models ─────────────────────────────────────────────
    video_tokenizer, lam, dynamics_model, video_idx_to_lat, action_idx_to_lat = \
        load_pretrained_models(args.config, device)

    # ── Initialize rollout state ────────────────────────────────
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

    # ── Seed from real data ─────────────────────────────────────
    print("Loading starting clip from dataset...")
    starting_clip, val_ds = get_starting_clip(cfg, vt_cfg, data_cfg, device)
    rollout.reset(starting_clip)
    print(f"Seeded context buffer with {cfg.context_length} real frames.")

    # Decode the last context frame for initial display
    with torch.no_grad():
        initial_pixels = video_tokenizer.decoder(rollout.latent_buffer[:, -1:])  # [1, 1, C, H, W]
    initial_frame = initial_pixels[0, 0]  # [C, H, W]
    initial_np = ((initial_frame + 1) / 2).clamp(0, 1).cpu().permute(1, 2, 0).numpy()

    # ── Build key map ───────────────────────────────────────────
    key_map = {
        getattr(pygame, k): v
        for k, v in ACTION_KEY_MAP_INDICES.items()
        if hasattr(pygame, k)
    }

    # ── Setup pygame ────────────────────────────────────────────
    native_size = vt_cfg.frame_size  # 64
    display_size = native_size * cfg.display_scale  # 512

    pygame.init()
    screen = pygame.display.set_mode((display_size, display_size))
    pygame.display.set_caption("MiniGenie — Interactive World Model")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    def display_frame(frame_np, info_text=""):
        """Render a frame (HWC, [0,1]) to the pygame window."""
        # Convert to uint8 and scale up with nearest-neighbor interpolation.
        # We use nearest-neighbor (not bilinear) because bilinear would visually
        # smooth over the model's actual blurriness, misrepresenting what it
        # really generated. Nearest-neighbor is honest about pixel-level output.
        frame_uint8 = (frame_np * 255).clip(0, 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(frame_uint8.swapaxes(0, 1))
        scaled = pygame.transform.scale(surface, (display_size, display_size))
        screen.blit(scaled, (0, 0))

        if info_text:
            text_surface = font.render(info_text, True, (255, 255, 0))
            screen.blit(text_surface, (5, 5))

        pygame.display.flip()

    # Show initial frame
    display_frame(initial_np, "Press W/A/S/D to act, R to reset, ESC to quit")

    # ── Main loop ───────────────────────────────────────────────
    frame_count = 0
    running = True
    clip_idx = 0  # for cycling through reset clips

    print()
    print("Controls:")
    print("  W/A/S/D  — action keys (placeholder mapping)")
    print("  SPACE    — idle/no-op action")
    print("  R        — reset to a new starting clip")
    print("  ESC / Q  — quit")
    print()

    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    # Reset to a new clip
                    clip_idx = (clip_idx + 1) % len(val_ds)
                    new_clip = val_ds[clip_idx].unsqueeze(0).to(device)
                    rollout.reset(new_clip)
                    frame_count = 0

                    with torch.no_grad():
                        reset_pixels = video_tokenizer.decoder(rollout.latent_buffer[:, -1:])
                    reset_np = ((reset_pixels[0, 0] + 1) / 2).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    display_frame(reset_np, f"RESET — clip #{clip_idx}")
                    print(f"Reset to clip #{clip_idx}")
                    continue
                elif event.key in key_map:
                    action = key_map[event.key]

        # If no action key was pressed this frame, check held keys
        if action is None:
            keys = pygame.key.get_pressed()
            for pg_key, act_idx in key_map.items():
                if keys[pg_key]:
                    action = act_idx
                    break

        # Only step the model when an action is available
        if action is not None:
            t_start = time.time()

            _, frame_np = rollout.step(action)

            t_elapsed = time.time() - t_start
            frame_count += 1
            fps = 1.0 / max(t_elapsed, 1e-6)
            info = f"Frame {frame_count} | {t_elapsed:.2f}s/frame ({fps:.1f} fps) | Action {action}"

            display_frame(frame_np, info)
            print(f"  Frame {frame_count}: action={action}, {t_elapsed:.2f}s")

        clock.tick(60)  # Cap polling rate (doesn't affect generation speed)

    pygame.quit()
    print(f"\nSession ended. Generated {frame_count} frames total.")


if __name__ == "__main__":
    main()
