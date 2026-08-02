from pathlib import Path
from typing import Callable
from typing import Type

import torch.nn as nn

from src.utils.utils import import_all_modules

_supported_models = {}


def register_model(name: str) -> Callable:
    def _register_model_cls(model_cls: Type[nn.Module]) -> Type[nn.Module]:
        assert name not in _supported_models, f"Can not register duplicate model: {name}"
        _supported_models[name] = model_cls
        return model_cls
    return _register_model_cls


def get_model(model_name: str) -> Type[nn.Module]:
    assert model_name in _supported_models, f"There is no model: {model_name}"
    return _supported_models[model_name]


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.model")
