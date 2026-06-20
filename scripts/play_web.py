"""
Gradio web interface for interactive play.
Usage:
    uv run python scripts/play_web.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.loaders.zelda_dataset import ZeldaDataset
from datasets.loaders.sonic_dataset import SonicDataset
from datasets.loaders.pole_position_dataset import PolePositionDataset
from inference.model_loader import load_pretrained_models
from inference.rollout import RolloutState

DATASET_CLASSES = {"zelda": ZeldaDataset, "sonic": SonicDataset, "pole_position": PolePositionDataset}

# ── Action Mapping ──────────────────────────────────────────────
ACTION_MAP = {
    "left": 0,
    "right": 1,
    "up": 2,
    "down": 3,
    "idle": 0,
}

# ── Globals ─────────────────────────────────────────────────────
video_tokenizer = None
lam = None
dynamics_model = None
rollout_state = None
val_ds = None
clip_idx = -1
device = None

def init_models(config_path="configs/rollout.yaml"):
    global video_tokenizer, lam, dynamics_model, rollout_state, val_ds, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device}...")
    
    video_tokenizer, lam, dynamics_model, video_idx_to_lat, action_idx_to_lat = \
        load_pretrained_models(config_path, device)
        
    cfg = OmegaConf.load(config_path)
    data_cfg = OmegaConf.load(cfg.data_config)
    vt_cfg = OmegaConf.load(cfg.video_tokenizer_config)
    
    rollout_state = RolloutState(
        video_tokenizer=video_tokenizer,
        dynamics_model=dynamics_model,
        action_index_to_latents_fn=action_idx_to_lat,
        video_index_to_latents_fn=video_idx_to_lat,
        context_length=cfg.context_length,
        num_maskgit_steps=cfg.num_maskgit_steps,
        temperature=cfg.temperature,
        schedule_k=cfg.schedule_k,
    )
    
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
    print("Initialization complete.")

def scale_frame(frame_np, scale=8):
    """Upscale frame via nearest neighbor interpolation to preserve pixel art."""
    frame_uint8 = (frame_np * 255).clip(0, 255).astype(np.uint8)
    scaled = np.repeat(np.repeat(frame_uint8, scale, axis=0), scale, axis=1)
    return scaled

def reset_env():
    """Seed the context buffer with a new validation clip."""
    global clip_idx
    clip_idx = (clip_idx + 1) % len(val_ds)
    new_clip = val_ds[clip_idx].unsqueeze(0).to(device)
    
    rollout_state.reset(new_clip)
    
    # Decode the last context frame to display
    with torch.no_grad():
        reset_pixels = video_tokenizer.decoder(rollout_state.latent_buffer[:, -1:])
    reset_np = ((reset_pixels[0, 0] + 1) / 2).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    
    return scale_frame(reset_np)

def step_env(action_name):
    """Step the world model forward with the given action."""
    action_idx = ACTION_MAP[action_name]
    _, frame_np = rollout_state.step(action_idx)
    return scale_frame(frame_np)


# ── Initialize ──────────────────────────────────────────────────
init_models()

# ── Gradio UI ───────────────────────────────────────────────────
with gr.Blocks(title="MiniGenie Web Demo") as demo:
    gr.Markdown("# MiniGenie Interactive World Model")
    gr.Markdown("Click an action button to generate the next frame. The models will autoregressively predict the future.")
    
    with gr.Row():
        output_image = gr.Image(label="Generated Frame", type="numpy", interactive=False)
        
    with gr.Row():
        btn_left = gr.Button("Left")
        btn_right = gr.Button("Right")
        btn_up = gr.Button("Up")
        btn_down = gr.Button("Down")
        btn_idle = gr.Button("Idle")
        
    with gr.Row():
        btn_reset = gr.Button("Reset (Load new seed clip)", variant="primary")

    # Wire up callbacks
    btn_left.click(fn=lambda: step_env("left"), outputs=output_image)
    btn_right.click(fn=lambda: step_env("right"), outputs=output_image)
    btn_up.click(fn=lambda: step_env("up"), outputs=output_image)
    btn_down.click(fn=lambda: step_env("down"), outputs=output_image)
    btn_idle.click(fn=lambda: step_env("idle"), outputs=output_image)
    
    btn_reset.click(fn=reset_env, outputs=output_image)
    
    # Auto-load first frame
    demo.load(fn=reset_env, outputs=output_image)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
