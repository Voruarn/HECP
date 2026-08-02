from pathlib import Path
from typing import Callable
from typing import Type

from src.datamodule.base import BaseDataModule
from src.utils.utils import import_all_modules

_supported_datamodules = {}


def register_datamodule(name: str) -> Callable:
    def register_datamodule_cls(datamodule_cls: Type[BaseDataModule]) -> Type[BaseDataModule]:
        assert name not in _supported_datamodules, f"Can not register duplicate datamodule: {name}"
        _supported_datamodules[name] = datamodule_cls
        return datamodule_cls
    return register_datamodule_cls


def get_datamodule(datamodule_name: str) -> Type[BaseDataModule]:
    assert datamodule_name in _supported_datamodules, f"There is no datamodule: {datamodule_name}"
    return _supported_datamodules[datamodule_name]


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.datamodule")
