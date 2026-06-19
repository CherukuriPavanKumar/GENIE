from datasets.loaders.base_dataset import FrameCacheDataset


class ZeldaDataset(FrameCacheDataset):
    """Zelda OOT 2D gameplay. Defaults match TinyWorlds: native ~60fps capture,
    skip-4 for an effective ~15fps, first 1000 frames are title/intro junk."""

    def __init__(
        self,
        h5_path: str = "datasets/zelda/zelda_frames.h5",
        seq_len: int = 4,
        frame_size: int = 64,
        frame_skip: int = 8,
        load_start_index: int = 1000,
        split: str = "train",
        train_frac: float = 0.9,
        in_memory: bool = True,
    ) -> None:
        super().__init__(
            h5_path=h5_path,
            seq_len=seq_len,
            frame_size=frame_size,
            frame_skip=frame_skip,
            load_start_index=load_start_index,
            split=split,
            train_frac=train_frac,
            in_memory=in_memory,
        )
