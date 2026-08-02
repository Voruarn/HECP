from pathlib import Path
from typing import Callable

from omegaconf import DictConfig
from torch.optim import Optimizer

from src.utils.utils import import_all_modules

_supported_schedulers = {}


def register_scheduler(name: str) -> Callable:
    def _register_scheduler_func(scheduler_func: Callable):
        assert name not in _supported_schedulers, f"Can not register duplicate scheduler: {name}"
        _supported_schedulers[name] = scheduler_func
        return scheduler_func

    return _register_scheduler_func


def get_scheduler(optimizer: Optimizer, scheduler_cfg: DictConfig, **kwargs) -> dict:
    scheduler_name = scheduler_cfg.name
    assert scheduler_name in _supported_schedulers, f"There is no scheduler: {scheduler_name}"
    scheduler = _supported_schedulers[scheduler_name](optimizer, scheduler_params=scheduler_cfg.params, **kwargs)
    scheduler["name"] = "lr"
    return scheduler


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.module.optimizer.scheduler")
