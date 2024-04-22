from torch.utils.data import Dataset
import torch
from typing import Tuple


class DummyDataset(Dataset):
    """An iterator that produces repeated dummy data.
    Args:
        data_sample: a data sample that should be produced at each step.
        batch_size: the batch size for storing.
        sample_count: number of `data` samples in the dummy dataset.
    """

    def __init__(self, data_shape: torch.Size, num_classes: int, sample_count: int) -> None:
        self._data_sample = torch.zeros(data_shape)
        self._class_sample = torch.zeros((num_classes,), dtype=torch.int64)
        self._sample_count = sample_count

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._data_sample, self._class_sample
