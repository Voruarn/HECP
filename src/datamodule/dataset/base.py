from abc import ABC
from abc import abstractmethod

from omegaconf import DictConfig
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Base class of Dataset."""

    data: list

    def __init__(self, file_paths: str) -> None:
        self.file_paths = file_paths
        self._prepare()

    @classmethod
    def from_config(cls, dataset_cfg: DictConfig) -> "BaseDataset":
        return cls(**dataset_cfg)

    @abstractmethod
    def _prepare(self) -> None:
        """Initialize variables."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Get item from dataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)
