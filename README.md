# MiniGenie

Small-scale, architecturally faithful reimplementation of Google's Genie.
Inspired by [TinyWorlds](https://github.com/AlmondGod/tinyworlds), not a copy of it.

## Status: Phase 2 -- Data Pipeline

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py --dataset zelda
python scripts/visualize_batch.py --dataset zelda
```

Run this on a **CPU** machine type (no GPU needed for this phase). If you're on
Lightning AI Studio, this is also where the dataset should live persistently --
switching that same Studio to a GPU machine later for tokenizer training won't
require re-downloading anything.

## Validation gate

`scripts/visualize_batch.py` saves `batch_check_<dataset>.png`. Consecutive
frames (left to right per row) should show smooth, small game-state deltas --
not frozen near-duplicates, not unrelated jump-cuts. If they look wrong, adjust
`frame_skip` in `configs/data.yaml`, not the Dataset class itself.

## Repo layout

```
datasets/loaders/   -- Dataset classes (h5-consuming only, no video preprocessing)
datasets/zelda/      -- downloaded .h5 lands here (gitignored)
datasets/sonic/
configs/              -- all tunables (resolution, seq_len, batch_size)
scripts/              -- download_data.py, visualize_batch.py
tokenizer/            -- Phase 3, empty for now
lam/                  -- Phase 4, empty for now
dynamics/             -- Phase 5, empty for now
inference/            -- Phase 6, empty for now
```

## Design notes vs TinyWorlds

- TinyWorlds' `VideoHDF5Dataset` bundles raw-mp4-to-h5 preprocessing (cv2) with
  h5-to-training-sample serving. We only need the latter -- the `.h5` files are
  downloaded prebuilt from `AlmondGod/tinyworlds` on HuggingFace. cv2 is not a
  dependency here.
- Train/val split is on by default (`train_frac` in config). TinyWorlds ships
  with this off by default (`disable_test_split=True`).
- Base class + thin per-game subclass pattern kept from TinyWorlds -- justified
  because Zelda and Sonic have genuinely different junk-frame lengths and
  native capture fps, not just cosmetic differences.
# GENIE
# GENIE
