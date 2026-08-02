from pathlib import Path
from typing import Callable
from typing import Type

import torch.optim as optim
import torch_optimizer

from src.utils.utils import import_all_modules

_supported_optimizers = {}


def register_optimizer(name: str) -> Callable:
    def _register_optimizer_cls(optimizer_cls: Type[optim.Optimizer]) -> Type[optim.Optimizer]:
        assert name not in _supported_optimizers, f"Can not register duplicate optimizer: {name}"
        return optimizer_cls
    return _register_optimizer_cls


def get_optimizer(optimizer_name: str) -> Type[optim.Optimizer]:
    try:
        return eval(f"optim.{optimizer_name}")
    except AttributeError:
        try:
            return eval(f"ops.{optimizer_name}")
        except AttributeError:
            try:
                return eval(f"torch_optimizer.{optimizer_name}")
            except AttributeError:
                assert optimizer_name in _supported_optimizers, f"There is no optimizer: {optimizer_name}"
                return _supported_optimizers[optimizer_name]


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.module.optimizer")
