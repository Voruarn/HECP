from pathlib import Path
from typing import Callable
from typing import Type

from src.module.metric.base import BaseMetric
from src.utils.utils import import_all_modules

_supported_metrics = {}


def register_metric(name: str) -> Callable:
    def register_metric_cls(metric_cls: Type[BaseMetric]) -> Type[BaseMetric]:
        assert name not in _supported_metrics, f"Can not register duplicate module: {name}"
        _supported_metrics[name] = metric_cls
        return metric_cls
    return register_metric_cls


def get_metric(metric_name: str) -> Type[BaseMetric]:
    assert metric_name in _supported_metrics, f"There is no metric: {metric_name}"
    return _supported_metrics[metric_name]


# Automatically import any python files.
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "src.module.metric")
