from pathlib import Path
from typing import Callable
from typing import Type

import torch.nn as nn

from src.utils.utils import import_all_modules

_supported_losses = {}


def register_loss(name: str) -> Callable:
    def _register_loss_cls(loss_cls: Type[nn.Module]) -> Type[nn.Module]:
        assert name not in _supported_losses, f"Can not register duplicate loss, {name}"
        _supported_losses[name] = loss_cls
        return loss_cls
    return _register_loss_cls


def get_loss(loss_name: str) -> Type[nn.Module]:
    assert loss_name in _supported_losses, f"There is no loss: {loss_name}"
    return _supported_losses[loss_name]


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.module.loss")
