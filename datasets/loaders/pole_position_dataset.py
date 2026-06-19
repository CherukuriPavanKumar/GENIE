from datasets.loaders.base_dataset import FrameCacheDataset


class PolePositionDataset(FrameCacheDataset):
    """Pole Position gameplay. Car racing game requiring quick reactions."""

    def __init__(
        self,
        h5_path: str = "datasets/pole_position/pole_position_frames.h5",
        seq_len: int = 4,
        frame_size: int = 64,
        frame_skip: int = 2,
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
