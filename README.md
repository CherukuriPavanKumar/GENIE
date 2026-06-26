<div align="center">
  <h1>🧞‍♂️ MiniGenie</h1>
  <p><strong>A minimal, interactive world model built from scratch.</strong></p>
  <p>Train AI to understand physics, discover latent actions from raw video, and generate playable interactive simulations.</p>

  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training Pipeline</a> •
  <a href="#interactive-play">Interactive Play</a>
</div>

---

## 🌍 Overview

MiniGenie is a lightweight implementation of an interactive world model, inspired by Google DeepMind's Genie. It learns to simulate a playable 2D game environment directly from raw, unlabeled video footage. 

Unlike traditional reinforcement learning that requires ground-truth actions, MiniGenie uses a **Latent Action Model (LAM)** to automatically discover control inputs from raw pixels. Once trained, you can interactively "play" the generated world model using your keyboard!

Currently configured and optimized for the **Pole Position** racing game.

## ✨ Features

- 🧠 **Action Discovery (LAM)**: Discovers discrete latent actions from unlabeled video using an FSQ-quantized bottleneck. Features custom **Entropy Regularization** to prevent action collapse.
- 👁️ **Video Tokenizer**: High-fidelity spatial-temporal feature extraction using a Space-Time Transformer (STT) backbone.
- 🕹️ **Dynamics Engine**: An autoregressive MaskGIT-style transformer that predicts the future state of the world based on your real-time action inputs.
- 🎮 **Interactive Play**: Run the model locally with a real-time `pygame` interface, or use the `gradio` web UI for cloud environments.

---

## 🚀 Installation

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast Python package management.

```bash
# Clone the repository
git clone https://github.com/yourusername/MiniGenie.git
cd MiniGenie

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio
uv pip install omegaconf matplotlib tqdm h5py
uv pip install pygame gradio
```

*(Note: If you are installing on Linux and `pygame` fails to build, install the SDL2 development headers: `sudo dnf install SDL2-devel` or `sudo apt-get install libsdl2-dev`).*

---

## 💽 Getting the Data

Download the training datasets (stored in highly compressed HDF5 format). By default, this downloads the Pole Position racing dataset:

```bash
uv run python scripts/download_data.py --dataset pole_position
```

---

## 🧠 Training Pipeline

MiniGenie is trained in three distinct phases. You must train the models in this exact order, as each phase depends on the previous one.

### Phase 1: Video Tokenizer
Trains the FSQ-VAE to compress raw 64x64 video frames into discrete tokens.
```bash
uv run python scripts/train_video_tokenizer.py
```

### Phase 2: Action Tokenizer (Latent Action Model)
Discovers the hidden action space (e.g., steering left, right, accelerating) by analyzing transitions between consecutive frames.
```bash
uv run python scripts/train_action_tokenizer.py
```
*Tip: You can use `uv run python tests/inspect_actions.py` to visually verify what each discovered action code does!*

### Phase 3: Dynamics Model
The "physics engine". Learns to predict the next frame's tokens given the past frames and the current action input.
```bash
uv run python scripts/train_dynamics.py
```

---

## 🎮 Interactive Play

Once all three models are trained, you can drop into the generated world and play it!

### 1. Local Play (PyGame)
If you are running on a local machine with a physical monitor, use the PyGame interface for a smooth, real-time experience:
```bash
uv run python scripts/play.py
```
- **W/A/S/D**: Move/Steer
- **Space**: Idle
- **R**: Reset to a new seed clip

### 2. Cloud/Browser Play (Gradio)
If you are training on a remote server (like Lightning AI Studio) without a display manager, spin up the web UI:
```bash
uv run python scripts/play_web.py
```
This will generate a public URL you can open in your browser to play the game via web buttons.

### 3. Headless GIF Generation
Want to generate a cool rollout to share on Twitter or LinkedIn? Use the headless simulator to feed a pre-defined sequence of actions and export a `.gif`:
```bash
uv run python scripts/simulate_rollout.py --actions 0 0 1 1 2 2 --output demo.gif --fps 10
```

---

## 🛠️ Configuration

All hyperparameters and model sizes are controlled via YAML configs in the `configs/` directory.

- `configs/data.yaml` - Context window and batch sizes.
- `configs/video_tokenizer.yaml` - STT capacity for the visual encoder.
- `configs/action_tokenizer.yaml` - LAM settings (adjust `entropy_weight` here).
- `configs/dynamics.yaml` - Dynamics capacity and MaskGIT decoding steps.

## 🙏 Acknowledgements
Inspired by DeepMind's [Genie paper](https://arxiv.org/abs/2402.15391) and the excellent architectural explorations in the [TinyWorlds](https://github.com/AlmondGod/tinyworlds) repository.
