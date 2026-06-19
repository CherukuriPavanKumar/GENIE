from datasets.loaders.base_dataset import FrameCacheDataset


class SonicDataset(FrameCacheDataset):
    """Sonic gameplay. Defaults match TinyWorlds: shorter junk-frame skip
    (100 vs Zelda's 1000), same target ~15fps."""

    def __init__(
        self,
        h5_path: str = "datasets/sonic/sonic_frames.h5",
        seq_len: int = 4,
        frame_size: int = 64,
        frame_skip: int = 8,
        load_start_index: int = 100,
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
