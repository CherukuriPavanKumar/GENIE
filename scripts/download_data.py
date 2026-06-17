"""
Download prebuilt gameplay frame caches from AlmondGod/tinyworlds on HuggingFace.
These are already preprocessed (decoded, resized, cached as .h5) -- we never
run cv2/video decoding ourselves.

Usage:
    python scripts/download_data.py --dataset zelda
    python scripts/download_data.py --dataset sonic
"""
import argparse
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download

REPO_ID = "AlmondGod/tinyworlds"
REPO_TYPE = "dataset"

FILENAMES = {
    "zelda": "zelda_frames.h5",
    "sonic": "sonic_frames.h5",
    "pong": "pong_frames.h5",
    "picodoom": "picodoom_frames.h5",
    "pole_position": "pole_position_frames.h5",
}

OUT_DIRS = {
    "zelda": "datasets/zelda",
    "sonic": "datasets/sonic",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=FILENAMES.keys(), default="zelda")
    args = parser.parse_args()

    filename = FILENAMES[args.dataset]
    out_dir = Path(OUT_DIRS.get(args.dataset, f"datasets/{args.dataset}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {REPO_ID} (multi-GB, this takes a while)...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=out_dir,
    )
    print(f"Saved to: {path}")

    # Ground truth, not assumption -- report exactly what's in the file.
    with h5py.File(path, "r") as f:
        print(f"Keys in h5: {list(f.keys())}")
        if "frames" in f:
            shape, dtype = f["frames"].shape, f["frames"].dtype
            print(f"frames shape: {shape}, dtype: {dtype}")
            n, h, w, c = shape
            print(f"-> {n} frames at native {h}x{w}, {c} channels")
            if (h, w) != (64, 64):
                print(
                    f"Native resolution is {h}x{w}, not 64x64 -- the Dataset "
                    f"class resizes via F.interpolate automatically, but if "
                    f"you want raw 64x64 set frame_size=64 in configs/data.yaml "
                    f"(already the default)."
                )


if __name__ == "__main__":
    main()
