from typing import Union

import torch
from omegaconf import DictConfig
from torchmetrics import AUROC
from torchmetrics import Accuracy
from torchmetrics import AveragePrecision
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from src.module.metric import register_metric
from src.module.metric.base import BaseClassificationMetric


@register_metric(name="classification")
class ClassificationMetric(BaseClassificationMetric):
    def __init__(
        self,
        accuracy: DictConfig,
        auroc: DictConfig,
        recall_at_threshold: DictConfig,
        precision_at_threshold: DictConfig,
        classes: list,
        **kwargs,
    ) -> None:
        super().__init__(
            accuracy=accuracy,
            auroc=auroc,
            recall_at_threshold=recall_at_threshold,
            precision_at_threshold=precision_at_threshold,
            classes=classes,
            **kwargs,
        )

    def _build_accuracy(self) -> None:
        for partition in ["train", "valid"]:
            if getattr(self.accuracy, f"active_in_{partition}"):
                getattr(self, f"{partition}_metrics")["accuracy"] = Accuracy(
                    task="multiclass",
                    num_classes=len(self.classes)
                )

    def _build_ap(self) -> None:
        for partition in ["train", "valid"]:
            if getattr(self.ap, f"active_in_{partition}"):
                getattr(self, f"{partition}_metrics")["ap"] = AveragePrecision(
                    task="multiclass",
                    num_classes=len(self.classes),
                    average=self.ap.average,
                )

    def _build_auroc(self) -> None:
        for partition in ["train", "valid"]:
            if getattr(self.auroc, f"active_in_{partition}"):
                getattr(self, f"{partition}_metrics")["auroc"] = AUROC(
                    task="multiclass",
                    num_classes=len(self.classes),
                    average=self.auroc.average,
                )

    def _build_recall_at_threshold(self) -> None:
        for partition in ["train", "valid"]:
            if getattr(self.recall_at_threshold, f"active_in_{partition}"):
                threshold = self.recall_at_threshold.threshold
                getattr(self, f"{partition}_metrics")["recall_at_threshold"] = BinaryRecall(
                    average=self.recall_at_threshold.average,
                    threshold=threshold
                )

    def _build_precision_at_threshold(self) -> None:
        for partition in ["train", "valid"]:
            if getattr(self.precision_at_threshold, f"active_in_{partition}"):
                threshold = self.precision_at_threshold.threshold
                getattr(self, f"{partition}_metrics")["precision_at_threshold"] = BinaryPrecision(
                    average=self.precision_at_threshold.average,
                    threshold=threshold
                )

    def update(self, probs: torch.Tensor, targets: torch.Tensor, is_train: bool = True, *args, **kwargs) -> None:
        partition = "train" if is_train else "valid"
        metrics = getattr(self, f"{partition}_metrics")
        for k in metrics:
            if k in ["recall_at_threshold", "precision_at_threshold"]:
                metrics[k].update(probs[:, 1], targets)
            else:
                metrics[k].update(probs, targets)

    def compute(self, is_train: bool = True) -> dict:
        partition = "train" if is_train else "valid"
        metrics = getattr(self, f"{partition}_metrics")

        res = {}
        for k in metrics:
            res[f"{partition}_{k}"] = metrics[k].compute()
            metrics[k].reset()
        return res

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> None:
        for k in self.train_metrics:
            self.train_metrics[k].set_dtype(dst_type)
        for k in self.valid_metrics:
            self.valid_metrics[k].set_dtype(dst_type)
