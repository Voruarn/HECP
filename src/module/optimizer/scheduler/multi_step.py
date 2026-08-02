from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from src.module.optimizer.scheduler import register_scheduler


@register_scheduler("multi_step")
def multi_step_scheduler(optimizer: Optimizer, scheduler_params: DictConfig, **kwargs) -> dict:
    scheduler = MultiStepLR(
        optimizer=optimizer,
        gamma=scheduler_params.gamma,
        milestones=scheduler_params.milestones,
    )

    return {
        "interval": "epoch",
        "scheduler": scheduler,
    }
