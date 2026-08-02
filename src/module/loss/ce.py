from typing import Optional

import torch
import torch.nn as nn

from src.module.loss import register_loss


@register_loss(name="ce")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
