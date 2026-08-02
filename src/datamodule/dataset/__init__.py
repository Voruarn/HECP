from pathlib import Path
from typing import Callable
from typing import Type

from src.datamodule.dataset.base import BaseDataset
from src.utils.utils import import_all_modules

_supported_datasets = {}


def register_dataset(name: str) -> Callable:
    def _register_dataset_cls(dataset_cls: Type[BaseDataset]) -> Type[BaseDataset]:
        assert name not in _supported_datasets, f"Can not register duplicate dataset: {name}"
        _supported_datasets[name] = dataset_cls
        return dataset_cls
    return _register_dataset_cls


def get_dataset(dataset_name: str) -> Type[BaseDataset]:
    assert dataset_name in _supported_datasets, f"There is no dataset: {dataset_name}"
    return _supported_datasets[dataset_name]


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.datamodule.dataset")
