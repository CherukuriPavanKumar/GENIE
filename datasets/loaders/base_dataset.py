"""
Base class for loading prebuilt gameplay frame caches (.h5) into fixed-length
training sequences.

This is intentionally a "consumer" of cached datasets, not a "producer" --
it never touches raw video. AlmondGod's TinyWorlds repo has a sibling
VideoHDF5Dataset that does cv2.VideoCapture -> frames -> .h5 once, up front.
We download that already-built .h5 directly from HuggingFace (see
scripts/download_data.py), so that conversion step never runs here.

H5 schema (matches AlmondGod/tinyworlds's published datasets):
    'frames': uint8 array, shape (N, H, W, C), RGB
"""
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FrameCacheDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        seq_len: int = 4,
        frame_size: Optional[int] = 64,
        frame_skip: int = 1,
        load_start_index: int = 0,
        split: str = "train",
        train_frac: float = 0.9,
        in_memory: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"{h5_path} not found. Run scripts/download_data.py first."
            )

        self.seq_len = seq_len
        self.frame_size = frame_size
        self.frame_skip = max(1, frame_skip)
        self.in_memory = in_memory
        self._h5_file = None  # lazily opened per-worker if in_memory=False

        with h5py.File(self.h5_path, "r") as f:
            if "frames" not in f:
                raise KeyError(
                    f"Expected key 'frames' in {h5_path}, found: {list(f.keys())}"
                )
            n_total, h, w, c = f["frames"].shape
            self.native_shape = (h, w, c)

            usable_start = min(load_start_index, n_total)
            split_idx = usable_start + int((n_total - usable_start) * train_frac)
            if split == "train":
                self.start, self.end = usable_start, split_idx
            elif split == "val":
                self.start, self.end = split_idx, n_total
            else:
                raise ValueError("split must be 'train' or 'val'")

            self.data = f["frames"][self.start:self.end] if in_memory else None

    def _read_range(self, lo: int, hi: int) -> np.ndarray:
        if self.in_memory:
            return self.data[lo:hi]
        if self._h5_file is None:  # opened once per DataLoader worker process
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file["frames"][self.start + lo : self.start + hi]

    def __len__(self) -> int:
        span = self.seq_len * self.frame_skip
        return max(0, (self.end - self.start) - span)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for length {len(self)}")

        hi = index + self.seq_len * self.frame_skip
        raw = self._read_range(index, hi)[:: self.frame_skip]  # [T, H, W, C] uint8
        if raw.shape[0] != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} frames, got {raw.shape[0]}")

        frames = torch.from_numpy(raw.copy()).float().permute(0, 3, 1, 2)  # [T,C,H,W]

        if self.frame_size is not None and frames.shape[-1] != self.frame_size:
            frames = torch.nn.functional.interpolate(
                frames,
                size=(self.frame_size, self.frame_size),
                mode="bilinear",
                align_corners=False,
            )

        return (frames / 127.5) - 1.0  # [0, 255] -> [-1, 1]

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()
