from abc import ABC
from abc import abstractmethod
from typing import Union

import torch.nn as nn
from omegaconf import DictConfig


class BaseMetric(nn.Module, ABC):
    train_metrics: nn.ModuleDict
    valid_metrics: nn.ModuleDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.build()

    @abstractmethod
    def build(self) -> None:
        """Build metrics for the train and valid based on the given config."""
        raise NotImplementedError

    @abstractmethod
    def update(self, is_train: bool, *args, **kwargs) -> None:
        """Override this method to update metrics."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, is_train: bool = True) -> dict:
        """Return computed metrics and reset the metrics."""
        raise NotImplementedError
    

class BaseClassificationMetric(BaseMetric, ABC):
    def __init__(
        self,
        accuracy: DictConfig,
        auroc: DictConfig,
        recall_at_threshold: DictConfig,
        precision_at_threshold: DictConfig,
        classes: Union[dict, list],
        **kwargs,
    ) -> None:
        self.accuracy = accuracy
        self.auroc = auroc
        self.recall_at_threshold = recall_at_threshold
        self.precision_at_threshold = precision_at_threshold
        self.classes = classes
        super().__init__()

    def build(self) -> None:
        self.train_metrics = nn.ModuleDict()
        self.valid_metrics = nn.ModuleDict()
        if isinstance(self.classes, dict):
            for k in self.classes:
                self.train_metrics[k] = nn.ModuleDict()
                self.valid_metrics[k] = nn.ModuleDict()

        self._build_accuracy()
        self._build_auroc()
        self._build_recall_at_threshold()
        self._build_precision_at_threshold()

    @abstractmethod
    def _build_accuracy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _build_ap(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _build_auroc(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _build_recall_at_threshold(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _build_precision_at_threshold(self) -> None:
        raise NotImplementedError
